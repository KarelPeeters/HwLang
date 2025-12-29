use hwl_util::constants::HWL_FILE_EXTENSION;
use hwl_util::io::{IoErrorWithPath, recurse_for_each_file};
use hwl_util::swriteln;
use itertools::Itertools;
use std::ffi::OsStr;
use std::path::Path;

fn main() {
    // collect std sources
    let std_folder = "./src/std";
    println!("cargo:rerun-if-changed={std_folder}");

    let mut f = String::new();
    swriteln!(f, "&[");

    let mut found_any = false;

    recurse_for_each_file(Path::new(&std_folder), |steps, entry_path| {
        if entry_path.extension() == Some(OsStr::new(HWL_FILE_EXTENSION)) {
            let mut steps = steps.iter().map(|s| s.to_str().unwrap().to_owned()).collect_vec();
            steps.insert(0, "std".to_owned());
            steps.push(entry_path.file_stem().unwrap().to_str().unwrap().to_owned());

            let mut path = steps.join("/");
            path.push_str(".kh");

            let entry_path = entry_path.to_str().unwrap();
            let include_str = format!("include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", {entry_path:?})");

            swriteln!(f, "    StdSourceFile {{");
            swriteln!(f, "        path: {path:?},");
            swriteln!(f, "        steps: &{steps:?},");
            swriteln!(f, "        content: {include_str}),");
            swriteln!(f, "    }},");

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

    swriteln!(f, "]");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    std::fs::write(out_dir.join("std_source_files.in"), f).unwrap();
}
