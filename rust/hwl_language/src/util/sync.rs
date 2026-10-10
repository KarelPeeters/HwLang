use crate::util::Never;
use dashmap::{DashMap, Entry};
use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use unwrap_match::unwrap_match;

// TODO rename to ComputeOnceMap
pub struct ComputeOnceArena<K, V, S> {
    mutex: Mutex<()>,
    map: DashMap<K, Box<ItemInfo<K, V, S>>>,
}

unsafe impl<K: Send + Sync, V: Send + Sync, S: Send + Sync> Sync for ComputeOnceArena<K, V, S> {}

#[derive(Debug)]
struct ItemInfo<K, V, S> {
    done_atomic: AtomicBool,
    done_var: Condvar,
    state: UnsafeCell<ItemState<K, V, S>>,
}

#[derive(Debug)]
enum ItemState<K, V, S> {
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

impl<K: Debug + Copy + Hash + Eq, V: Debug, S: Debug + Clone> ComputeOnceArena<K, V, S> {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        ComputeOnceArena {
            mutex: Mutex::new(()),
            map: DashMap::with_shard_amount(dashmap_shard_count(thread_count).get()),
        }
    }

    /// Offer to compute an item, without caring or waiting for the result if someone else is already computing it.
    pub fn offer_to_compute(&self, item: K, f_compute: impl FnOnce() -> V) {
        // fast path without any exclusive locking
        if self.map.contains_key(&item) {
            return;
        }

        // try claiming the entry
        let guard = self.mutex.lock().unwrap();
        match self.map.entry(item) {
            Entry::Occupied(_) => return,
            Entry::Vacant(entry) => {
                entry.insert(Box::new(ItemInfo::new_progress()));
            }
        }

        // actually do something
        match self.impl_unvisited::<Never>(None, item, f_compute, |_| unreachable!(), guard) {
            Ok(_) => {}
            Err(never) => never.unreachable(),
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
        // fast path without any exclusive locking
        if let Some(entry) = self.map.get(&item)
            && entry.done_atomic.load(Ordering::Acquire)
        {
            let state = unsafe { &*entry.state.get() };
            let value = unwrap_match!(state, ItemState::Done(v) => v);
            return Ok(value);
        }

        // try claiming the entry
        let mut guard = self.mutex.lock().unwrap();
        enum Next {
            CheckCyclesAndWait,
            DoComputation,
        }
        let next = match self.map.entry(item) {
            Entry::Occupied(entry) => {
                if entry.get().done_atomic.load(Ordering::Acquire) {
                    let state = unsafe { &*entry.get().state.get() };
                    let value = unwrap_match!(state, ItemState::Done(v) => v);
                    return Ok(value);
                }

                Next::CheckCyclesAndWait
            }
            Entry::Vacant(entry) => {
                entry.insert(Box::new(ItemInfo::new_progress()));
                Next::DoComputation
            }
        };

        match next {
            Next::CheckCyclesAndWait => {
                // check for cycles
                if let Some((origin_item, origin_path)) = origin {
                    check_cycle(&guard, &self.map, item, origin_item, &origin_path).map_err(f_cycle)?;

                    // set dependency of origin item to current item
                    // safety: we are the thread that is computing the origin item, so no one else will be modifying its state
                    let origin_state = self.map.get(&origin_item).unwrap().state.get();
                    let origin_state = unsafe { &mut *origin_state };

                    match origin_state {
                        ItemState::Progress(dep) => {
                            // TODO re-enable this assert or something similar to it?
                            // assert_eq!(dep.as_ref().map(|d| &d.next), Some(&item));
                            *dep = Some(Dependency {
                                next: item,
                                path: origin_path,
                            })
                        }
                        ItemState::Done(_) => unreachable!(),
                    }
                }

                // wait until done
                let (cond_var, state_ptr) = {
                    let entry = self.map.get(&item).unwrap();
                    let entry_ptr = entry.as_ref() as *const ItemInfo<_, _, _>;
                    let entry_ref = unsafe { &*entry_ptr };
                    (&entry_ref.done_var, entry_ref.state.get())
                };
                loop {
                    guard = cond_var.wait(guard).unwrap();
                    match unsafe { &*state_ptr } {
                        ItemState::Progress(_) => continue,
                        ItemState::Done(result) => break Ok(result),
                    }
                }
            }
            Next::DoComputation => self.impl_unvisited(origin, item, f_compute, f_cycle, guard),
        }
    }

    fn impl_unvisited<E>(
        &self,
        origin: Option<(K, Vec<S>)>,
        item: K,
        f_compute: impl FnOnce() -> V,
        f_cycle: impl FnOnce(Vec<&S>) -> E,
        guard: MutexGuard<()>,
    ) -> Result<&V, E> {
        if let Some((prev_item, path)) = origin {
            // unit cycle, report
            if prev_item == item {
                return Err(f_cycle(path.iter().collect()));
            }

            // set dependency of origin item to current item
            let prev_stat_ptr = self.map.get(&prev_item).unwrap().state.get();
            match unsafe { &mut *prev_stat_ptr } {
                ItemState::Progress(prev_dependency) => *prev_dependency = Some(Dependency { next: item, path }),
                ItemState::Done(_) => unreachable!(),
            }
        }

        drop(guard);

        // do the computation
        let result = f_compute();

        // reacquire the lock and mark as done, notifying any waiters
        let guard = self.mutex.lock();
        let entry = self.map.get(&item).unwrap();
        {
            let state = unsafe { &mut *entry.state.get() };
            assert!(matches!(*state, ItemState::Progress(_)));
            *state = ItemState::Done(result);
            entry.done_atomic.store(true, Ordering::Release);
            drop(guard);
        }
        entry.done_var.notify_all();

        // get a reference to the result we just stored, to return to the caller
        let state = unsafe { &*entry.state.get() };
        match state {
            ItemState::Progress(_) => unreachable!(),
            ItemState::Done(result) => Ok(result),
        }
    }
}

fn check_cycle<'a, K: Debug + Eq + Hash + Copy, V, S: Debug>(
    _guard: &MutexGuard<()>,
    map: &'a DashMap<K, Box<ItemInfo<K, V, S>>>,
    item: K,
    origin_item: K,
    origin_path: &'a Vec<S>,
) -> Result<(), Vec<&'a S>> {
    // TODO avoid allocating this vec if there is no cycle?
    let mut curr = item;
    let mut full_path = vec![];

    loop {
        if curr == origin_item {
            full_path.extend(origin_path);
            return Err(full_path);
        }

        let state = &map.get(&curr).unwrap().state;
        curr = match unsafe { &*state.get() } {
            ItemState::Done(_) | ItemState::Progress(None) => break,
            ItemState::Progress(Some(dependency)) => {
                let &Dependency { next, ref path } = dependency;
                full_path.extend(path);
                next
            }
        };
    }

    Ok(())
}

impl<K, V, S> ItemInfo<K, V, S> {
    fn new_progress() -> Self {
        ItemInfo {
            done_atomic: AtomicBool::new(false),
            done_var: Condvar::new(),
            state: UnsafeCell::new(ItemState::Progress(None)),
        }
    }
}

impl<K: Debug, S: Debug> Debug for Dependency<K, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dependency")
            .field("next", &self.next)
            .field("path", &"...")
            .finish()
    }
}

pub struct ComputeOnceMap<K, V> {
    inner: DashMap<K, Box<OnceLock<V>>>,
}

#[derive(Debug)]
pub struct KeyAlreadyPresent;

impl<K: Eq + Hash, V> ComputeOnceMap<K, V> {
    pub fn new() -> Self {
        Self { inner: DashMap::new() }
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let ptr: *const OnceLock<V> = &**self.inner.get(k)?;
        // Entries are boxed and never removed: releasing the shard lock preserves the pointee.
        unsafe { &*ptr }.get()
    }

    pub fn set(&self, k: K, v: V) -> Result<&V, KeyAlreadyPresent> {
        match self.inner.entry(k) {
            Entry::Occupied(_) => Err(KeyAlreadyPresent),
            Entry::Vacant(entry) => {
                let ptr: *const OnceLock<V> = &**entry.insert(Box::new(OnceLock::from(v)));
                Ok(unsafe { &*ptr }.get().unwrap())
            }
        }
    }

    pub fn get_or_compute(&self, k: K, f: impl FnOnce(&K) -> V) -> &V
    where
        K: Clone,
    {
        let entry = self.inner.entry(k.clone()).or_insert_with(|| Box::new(OnceLock::new()));
        let ptr: *const OnceLock<V> = &**entry;
        drop(entry);
        // Recursive computations can access another key in the same shard. Release the
        // DashMap guard first; OnceLock still ensures one computation per key.
        unsafe { &*ptr }.get_or_init(|| f(&k))
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
        {
            let mut inner = self.mutex.lock().unwrap();
            inner.queue.push_back(item);

            // TODO this weird behavior was created for the python API, think about a more correct solution
            if inner.done {
                inner.done = false;
                inner.waiting_count = 0;
            }
        }

        self.cond.notify_one();
    }

    pub fn push_batch(&self, items: impl IntoIterator<Item = T>) {
        let delta = {
            let mut inner = self.mutex.lock().unwrap();

            let len_before = inner.queue.len();
            inner.queue.extend(items);
            inner.queue.len() - len_before
        };

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
        let mut inner = self.mutex.lock().unwrap();

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
            inner = self.cond.wait(inner).unwrap();
            inner.waiting_count -= 1;
        }
    }
}

pub fn dashmap_shard_count(thread_count: NonZeroUsize) -> NonZeroUsize {
    // matches the default shard count of DashMap
    // TODO try tuning this
    NonZeroUsize::new((thread_count.get() * 4).next_power_of_two()).unwrap()
}

#[cfg(test)]
mod compute_once_map_tests {
    use super::ComputeOnceMap;
    use std::hash::{Hash, Hasher};

    #[derive(Clone, PartialEq, Eq)]
    struct SameShard(u32);

    impl Hash for SameShard {
        fn hash<H: Hasher>(&self, state: &mut H) {
            0u32.hash(state);
        }
    }

    #[test]
    fn recursive_computation_in_same_shard() {
        let map = ComputeOnceMap::new();
        let value = map.get_or_compute(SameShard(0), |_| *map.get_or_compute(SameShard(1), |_| 41) + 1);
        assert_eq!(*value, 42);
        assert_eq!(*map.get_or_compute(SameShard(0), |_| panic!("computed twice")), 42);
    }

    #[test]
    fn concurrent_computation_happens_once() {
        let map = ComputeOnceMap::new();
        let count = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    assert_eq!(
                        *map.get_or_compute(0, |_| {
                            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            42
                        }),
                        42
                    );
                });
            }
        });
        assert_eq!(count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
}
