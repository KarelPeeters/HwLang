use hwl_util::constants::COMPILE_THREAD_STACK_SIZE;
use itertools::Itertools;
use std::num::NonZeroUsize;
use std::sync::Mutex;

pub struct ThreadPool {
    thread_count: NonZeroUsize,
    pool: yastl::Pool,
}

impl ThreadPool {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        let config = yastl::ThreadConfig::new()
            .prefix("compile")
            .stack_size(COMPILE_THREAD_STACK_SIZE);
        let pool = yastl::Pool::with_config(thread_count.get(), config);
        ThreadPool { pool, thread_count }
    }

    pub fn thread_count(&self) -> NonZeroUsize {
        self.thread_count
    }

    /// Run `f` on each thread in the pool, collecting results in order.
    pub fn broadcast<T: Send>(&self, f: impl Fn() -> T + Sync) -> Vec<T> {
        let slots: Vec<Mutex<Option<T>>> = (0..self.thread_count.get()).map(|_| Mutex::new(None)).collect_vec();
        self.pool.scoped(|scope| {
            for slot in &slots {
                scope.execute(|| {
                    *slot.lock().unwrap() = Some(f());
                });
            }
        });
        slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
    }
}
