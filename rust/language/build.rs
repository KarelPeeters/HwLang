use lalrpop::Configuration;

fn main() {
    std::thread::Builder::new()
        .stack_size(1024*1024*1024)
        .spawn(|| {
            Configuration::new().use_cargo_dir_conventions().process_file("src/syntax/grammar.lalrpop").unwrap()
        })
        .unwrap().join().unwrap();
}
