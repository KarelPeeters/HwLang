fn main() {
    // compile grammar, with increased stack space to avoid stack overflows
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(|| {
            lalrpop::Configuration::new()
                .use_cargo_dir_conventions()
                .process_file("src/syntax/grammar.lalrpop")
                .unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
