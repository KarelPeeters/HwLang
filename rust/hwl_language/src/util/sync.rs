use std::collections::VecDeque;

use parking_lot::{Condvar, Mutex};

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
        self.inner.get_or_init(f);
    }
}

// TODO is there a way to avoid the single mutex bottleneck?
pub struct SharedQueue<T> {
    thread_count: usize,
    mutex: Mutex<SharedQueueInner<T>>,
    cond: Condvar,
}

struct SharedQueueInner<T> {
    queue: VecDeque<T>,
    waiting_count: usize,
    done: bool,
}

impl<T> SharedQueue<T> {
    pub fn new(thread_count: usize) -> Self {
        assert!(thread_count > 0);
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
            if inner.waiting_count == self.thread_count {
                inner.done = true;
                self.cond.notify_all();
                return None;
            }
            self.cond.wait(&mut inner);
            inner.waiting_count -= 1;
        }
    }
}
