fn main() {
    // increase stack space for lalrpop
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(main_inner)
        .unwrap()
        .join()
        .unwrap();
}

fn main_inner() {
    // compile grammar
    lalrpop::Configuration::new()
        .use_cargo_dir_conventions()
        .process_file("src/syntax/grammar.lalrpop")
        .unwrap();
}
