#![no_main]

use hwl_language::syntax::source::FileId;
use hwl_language::syntax::token::tokenize;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (&str, bool)| target(data));

fn target(data: (&str, bool)) {
    let (s, i) = data;
    let _ = tokenize(FileId::dummy(), s, i);
}
