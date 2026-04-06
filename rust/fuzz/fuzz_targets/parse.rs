#![no_main]

use hwl_common::source::FileId;
use hwl_language::syntax::parser::parse_file_content_without_recovery;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| target(data));

fn target(data: &str) {
    let _ = parse_file_content_without_recovery(FileId::dummy(), data);
}
