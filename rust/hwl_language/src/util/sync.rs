use once_map::OnceMap;
use parking_lot::{Condvar, Mutex};
use std::hash::Hash;
use std::{collections::VecDeque, num::NonZeroUsize};

pub struct ComputeOnce<T> {
    inner: std::sync::OnceLock<T>,
}

// TODO add some cycle deadlock detection, what if we're already the thread that is computing this,
//   but then we're asking for it again?
impl<T> ComputeOnce<T> {
    pub fn new() -> Self {
        Self {
            inner: std::sync::OnceLock::new(),
        }
    }

    pub fn into_inner(self) -> Option<T> {
        self.inner.into_inner()
    }

    pub fn get_or_compute(&self, f: impl FnOnce() -> T) -> &T {
        self.inner.get_or_init(f)
    }

    pub fn offer_to_compute(&self, f: impl FnOnce() -> T) -> () {
        // TODO better implementation that does not block if someone else is already computing
        //   (we might need a fully custom implementation of OnceLock)
        // TODO RwLock probably works pretty well, benchmark for fun
        self.inner.get_or_init(f);
    }
}

// TODO cycle detection with nice error messages (also needed to prevent deadlocks)
// TODO implement with better concurrency primitives
pub struct ComputeOnceMap<K, V> {
    inner: OnceMap<K, Box<V>>,
}

impl<K: Hash + Eq, V> ComputeOnceMap<K, V> {
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
