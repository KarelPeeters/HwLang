use hwl_util::constants::LANGUAGE_FILE_EXTENSION;
use hwl_util::io::{recurse_for_each_file, IoErrorWithPath};
use hwl_util::swriteln;
use itertools::Itertools;
use std::ffi::OsStr;
use std::path::Path;
// TODO move the things used from hwl_language into a separate crate,
//   so the cargo build does not appear to fail whenever _anything_ in hwl_language fails to build

fn main() {
    // collect stdlib
    // TODO move std to a better location
    // TODO add options to binary to choose builtin vs external std?
    // TODO make this configurable, so we don't have to re-built the compiler on every stdlib change
    let std_folder = "../../design/project/std";
    println!("cargo:rerun-if-changed={}", std_folder);

    let mut f = String::new();
    swriteln!(f, "pub const STD_SOURCES: &[(&[&str], &str, &str)] = &[");

    let mut found_any = false;

    recurse_for_each_file(Path::new(&std_folder), |steps, entry| {
        let entry_path = entry.path();
        if entry_path.extension() == Some(OsStr::new(LANGUAGE_FILE_EXTENSION)) {
            let mut steps = steps.iter().map(|s| s.to_str().unwrap().to_owned()).collect_vec();
            steps.insert(0, "std".to_owned());
            steps.push(entry_path.file_stem().unwrap().to_str().unwrap().to_owned());

            let steps_str = steps.join("/");

            let entry_path = entry_path.to_str().unwrap();
            let include_str = format!("include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", {entry_path:?})");

            swriteln!(f, "    (&{steps:?}, {steps_str:?}, {include_str})),",);
            found_any = true;
        }
        Ok::<_, IoErrorWithPath>(())
    })
    .unwrap();

    if !found_any {
        panic!(
            "No files found in the stdlib folder. Path is {std_folder} which resolves to {:?}",
            std::env::current_dir().unwrap().join(std_folder),
        );
    }

    swriteln!(f, "];");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    std::fs::write(out_dir.join("std_sources.rs"), f).unwrap();
}
