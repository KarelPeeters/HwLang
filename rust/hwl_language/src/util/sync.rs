use std::collections::VecDeque;

use parking_lot::Mutex;

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

pub struct SharedQueue<T> {
    inner: Mutex<VecDeque<T>>,
}

impl<T> SharedQueue<T> {
    pub fn new(workers: usize) -> Self {
        if workers != 1 {
            todo!("support multiple workers in SharedQueue");
        }
        Self {
            inner: Mutex::new(VecDeque::new()),
        }
    }

    pub fn push(&self, item: T) {
        self.inner.lock().push_back(item);
    }

    pub fn push_batch(&self, items: impl IntoIterator<Item = T>) {
        self.inner.lock().extend(items);
    }

    pub fn pop(&self) -> Option<T> {
        // for now we know that there is just a single thread (see the todo in the constructor),
        //   so if this is empty we are sure that no future elements will be added
        self.inner.lock().pop_front()
    }
}
