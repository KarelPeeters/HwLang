#![no_main]

use hwl_language::syntax::source::FileId;
use hwl_language::syntax::token::tokenize;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| target(data));

fn target(data: &str) {
    let _ = tokenize(FileId::dummy(), data);
}
