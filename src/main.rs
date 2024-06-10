use clap::Parser;
use std::path::PathBuf;
use itertools::Itertools;
use hwlang::resolve::compile_set::CompileSet;
use hwlang::util::recurse_for_each_file;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
}

fn main() {
    let args = Args::parse();

    let mut set = CompileSet::new();

    recurse_for_each_file(&args.root, &mut |stack, f| {
        println!("compiling {:?}", f.path());
        let source = std::fs::read_to_string(f.path()).unwrap();
        let stack = stack.iter().map(|s| s.to_str().unwrap().to_owned()).collect_vec();
        set.add_file(stack, source).unwrap();
    }).unwrap();

    let _set_checked = set.check();
}
