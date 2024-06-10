use clap::Parser;
use std::path::PathBuf;
use hwlang::util::recurse_for_each_file;
use hwlang::syntax::parse_package_content;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    recurse_for_each_file(&args.root, &mut |f| {
        let src = std::fs::read_to_string(f.path()).unwrap();
        parse_package_content(&src).unwrap();
    }).unwrap();
}
