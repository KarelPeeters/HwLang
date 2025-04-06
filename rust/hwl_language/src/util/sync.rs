use crate::util::Never;
use indexmap::IndexMap;
use once_map::OnceMap;
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::fmt::Debug;
use std::hash::Hash;
use std::{collections::VecDeque, num::NonZeroUsize};

pub struct ComputeOnceArena<K, V, S> {
    inner: Mutex<ComputeOnceArenaInner<K, V, S>>,
}

struct ComputeOnceArenaInner<K, V, S> {
    map: IndexMap<K, ItemInfo<K, V, S>>,
}

#[derive(Debug)]
struct ItemInfo<K, V, S> {
    done_var: Condvar,
    state: ItemState<K, V, S>,
}

#[derive(Debug)]
enum ItemState<K, V, S> {
    /// This item has never been visited before.
    Unvisited,
    /// Computation of this item is in progress.
    /// This stores another item this computation depends on if any, and the path between them.
    Progress(Option<Dependency<K, S>>),
    /// This item has been fully computed.
    Done(V),
}

struct Dependency<K, S> {
    next: K,
    // TODO we could use a reference for this,
    //   we know it won't be dropped or moved before the outer call returns
    path: Vec<S>,
}

impl<K: Debug, S: Debug> Debug for Dependency<K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dependency")
            .field("next", &self.next)
            .field("path", &"...")
            .finish()
    }
}

impl<K: Debug + Copy + Hash + Eq, V: Debug, S: Debug + Clone> ComputeOnceArena<K, V, S> {
    pub fn new(keys: impl IntoIterator<Item = K>) -> Self {
        let map = keys
            .into_iter()
            .map(|k| {
                let info = ItemInfo {
                    done_var: Condvar::new(),
                    state: ItemState::Unvisited,
                };
                (k, info)
            })
            .collect();
        ComputeOnceArena {
            inner: Mutex::new(ComputeOnceArenaInner { map }),
        }
    }

    /// Offer to compute an item, without caring or waiting for the result if someone else is already computing it.
    pub fn offer_to_compute(&self, item: K, f_compute: impl FnOnce() -> V) {
        let guard = self.inner.lock();
        let item_map_index = guard.map.get_index_of(&item).unwrap();

        match guard.map[item_map_index].state {
            ItemState::Unvisited => {
                match self.impl_unvisited::<Never>(None, item, f_compute, |_| unreachable!(), guard, item_map_index) {
                    Ok(_) => {}
                    Err(never) => never.unreachable(),
                }
            }
            ItemState::Progress(_) | ItemState::Done(_) => {
                // someone else is already computing this item or it has already been computed,
                //   we don't care about the result so we can return immediately
            }
        }
    }

    /// Get the result of a computation, computing it or waiting for someone else to compute it if necessary.
    pub fn get_or_compute<E>(
        &self,
        origin: Option<(K, Vec<S>)>,
        item: K,
        f_compute: impl FnOnce() -> V,
        f_cycle: impl FnOnce(Vec<&S>) -> E,
    ) -> Result<&V, E> {
        // TODO add fast path for common "done" case that avoids taking this single global lock
        let mut guard = self.inner.lock();
        let item_map_index = guard.map.get_index_of(&item).unwrap();

        match &guard.map[item_map_index].state {
            ItemState::Unvisited => self.impl_unvisited(origin, item, f_compute, f_cycle, guard, item_map_index),
            ItemState::Progress(_) => {
                // check for cycles
                if let Some((origin_item, origin_path)) = origin {
                    check_cycle(&guard.map, item, origin_item, &origin_path).map_err(f_cycle)?;

                    // set dependency of origin item to current item
                    match &mut guard.map.get_mut(&origin_item).unwrap().state {
                        ItemState::Progress(prev_dependency) => {
                            *prev_dependency = Some(Dependency {
                                next: item,
                                path: origin_path,
                            })
                        }
                        ItemState::Unvisited | ItemState::Done(_) => unreachable!(),
                    }
                }

                // wait until done
                let done_var = unsafe { extend_lifetime(self, &guard.map[item_map_index].done_var) };
                loop {
                    done_var.wait(&mut guard);
                    match &guard.map[item_map_index].state {
                        ItemState::Unvisited => unreachable!(),
                        ItemState::Progress(_) => continue,
                        ItemState::Done(result) => {
                            let result = unsafe { extend_lifetime(self, result) };
                            break Ok(result);
                        }
                    }
                }
            }
            ItemState::Done(result) => {
                // simple return
                let result = unsafe { extend_lifetime(self, result) };
                Ok(result)
            }
        }
    }

    fn impl_unvisited<E>(
        &self,
        origin: Option<(K, Vec<S>)>,
        item: K,
        f_compute: impl FnOnce() -> V,
        f_cycle: impl FnOnce(Vec<&S>) -> E,
        mut guard: MutexGuard<ComputeOnceArenaInner<K, V, S>>,
        item_map_index: usize,
    ) -> Result<&V, E> {
        if let Some((prev_item, path)) = origin {
            // unit cycle, report
            if prev_item == item {
                return Err(f_cycle(path.iter().collect()));
            }

            // set dependency of origin item to current item
            match &mut guard.map.get_mut(&prev_item).unwrap().state {
                ItemState::Progress(prev_dependency) => *prev_dependency = Some(Dependency { next: item, path }),
                ItemState::Unvisited | ItemState::Done(_) => unreachable!(),
            }
        }

        // TODO avoid duplicate map lookup, we already did this to initially figure out we are unvisited
        // set current state to in progress and release the lock, so other computations can happen in parallel
        guard.map[item_map_index].state = ItemState::Progress(None);
        drop(guard);

        // do the computation
        let result = f_compute();

        // reacquire the lock and mark as done, notifying any waiters
        let mut guard = self.inner.lock();
        let slot = &mut guard.map[item_map_index];
        assert!(matches!(&slot.state, ItemState::Progress(_)));
        slot.state = ItemState::Done(result);
        slot.done_var.notify_all();

        // get a reference to the result we just stored, to return to the caller
        match &slot.state {
            ItemState::Unvisited | ItemState::Progress(_) => unreachable!(),
            ItemState::Done(result) => {
                let result = unsafe { extend_lifetime(self, result) };
                Ok(result)
            }
        }
    }
}

fn check_cycle<'a, K: Debug + Eq + Hash + Copy, V, S>(
    map: &'a IndexMap<K, ItemInfo<K, V, S>>,
    start: K,
    origin_item: K,
    origin_path: &'a Vec<S>,
) -> Result<(), Vec<&'a S>> {
    if start == origin_item {
        return Err(origin_path.iter().collect());
    }

    // TODO avoid allocating this vec if there is no cycle?
    // TODO avoid duplicate initial map lookup?
    let mut full_path = vec![];

    let mut curr = start;
    loop {
        curr = match &map.get(&curr).unwrap().state {
            ItemState::Unvisited => unreachable!(),
            ItemState::Done(_) | ItemState::Progress(None) => break,
            ItemState::Progress(Some(dependency)) => {
                let &Dependency { next, ref path } = dependency;
                full_path.extend(path);

                if next == origin_item {
                    // TODO where to put this?
                    full_path.extend(origin_path);
                    return Err(full_path);
                }

                next
            }
        };
    }

    Ok(())
}

unsafe fn extend_lifetime<'a, 'b, A, B>(lifetime: &'a A, value: &'b B) -> &'a B {
    let _ = lifetime;
    std::mem::transmute::<&'b B, &'a B>(value)
}

pub struct ComputeOnceMap<K, V> {
    inner: OnceMap<K, Box<V>>,
}

impl<K: Debug + Hash + Eq, V> ComputeOnceMap<K, V> {
    pub fn new() -> Self {
        Self { inner: OnceMap::new() }
    }

    pub fn get_or_compute(&self, k: K, f: impl FnOnce(&K) -> V) -> &V {
        self.inner.insert(k, |k| Box::new(f(k)))
    }
}

// TODO is there a way to avoid the single mutex bottleneck?
pub struct SharedQueue<T> {
    thread_count: NonZeroUsize,
    mutex: Mutex<SharedQueueInner<T>>,
    cond: Condvar,
}

struct SharedQueueInner<T> {
    queue: VecDeque<T>,
    waiting_count: usize,
    done: bool,
}

impl<T> SharedQueue<T> {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        Self {
            thread_count,
            mutex: Mutex::new(SharedQueueInner {
                queue: VecDeque::new(),
                waiting_count: 0,
                done: false,
            }),
            cond: Condvar::new(),
        }
    }

    pub fn push(&self, item: T) {
        self.mutex.lock().queue.push_back(item);
        self.cond.notify_one();
    }

    pub fn push_batch(&self, items: impl IntoIterator<Item = T>) {
        let mut inner = self.mutex.lock();

        let len_before = inner.queue.len();
        inner.queue.extend(items);
        let delta = inner.queue.len() - len_before;

        match delta {
            0 => {}
            1 => {
                self.cond.notify_one();
            }
            _ => {
                self.cond.notify_all();
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut inner = self.mutex.lock();

        loop {
            if inner.done {
                return None;
            }
            if let Some(item) = inner.queue.pop_front() {
                return Some(item);
            }

            inner.waiting_count += 1;
            if inner.waiting_count == self.thread_count.get() {
                inner.done = true;
                self.cond.notify_all();
                return None;
            }
            self.cond.wait(&mut inner);
            inner.waiting_count -= 1;
        }
    }
}
