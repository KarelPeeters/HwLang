#![no_main]

use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::source::FileId;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| target(data));

fn target(data: &str) {
    let _ = parse_file_content(FileId::dummy(), data);
}
