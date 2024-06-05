fn main() {
    std::thread::Builder::new()
        .stack_size(1024*1024*1024)
        .spawn(|| {
            lalrpop::process_root().unwrap()
        })
        .unwrap().join().unwrap();
}
