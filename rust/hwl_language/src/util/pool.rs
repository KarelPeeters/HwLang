use hwl_util::constants::COMPILE_THREAD_STACK_SIZE;
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::thread::ScopedJoinHandle;

// TODO provide real implementation that actually reuses threads
pub struct ThreadPool {
    thread_count: NonZeroUsize,
}

pub struct Scope<'scope, 'env: 'scope> {
    scope: &'scope std::thread::Scope<'scope, 'env>,

    next_index: Cell<usize>,
}

impl ThreadPool {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        ThreadPool { thread_count }
    }

    pub fn thread_count(&self) -> NonZeroUsize {
        self.thread_count
    }

    pub fn scope<'env, F, R>(&self, f: F) -> R
    where
        F: for<'scope> FnOnce(Scope<'scope, 'env>) -> R,
    {
        std::thread::scope(|scope| {
            f(Scope {
                scope,
                next_index: Cell::new(0),
            })
        })
    }
}

impl<'scope, 'env> Scope<'scope, 'env> {
    pub fn spawn<F, T>(&self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        let index = self.next_index.get();
        self.next_index.set(index + 1);

        std::thread::Builder::new()
            .name(format!("compile-{index}"))
            .stack_size(COMPILE_THREAD_STACK_SIZE)
            .spawn_scoped(self.scope, f)
            .unwrap()
    }
}
