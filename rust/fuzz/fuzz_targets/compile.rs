#![no_main]

use hwl_language::front::compile::{compile, ElaborationSet};
use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::front::print::NoPrintHandler;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FilePath, SourceDatabaseBuilder};
use hwl_language::util::NON_ZERO_USIZE_ONE;
use libfuzzer_sys::fuzz_target;

// TODO we can be a lot smarter here, generate only valid asts, reuse identifiers from a limited pool, ...
fuzz_target!(|data: String| target(data));

fn target(data: String) {
    let mut source = SourceDatabaseBuilder::new();
    source
        .add_file(
            FilePath(vec!["dummy".to_string()]),
            "dummy.kh".to_string(),
            data.to_string(),
        )
        .unwrap();
    let source = source.finish();

    let diags = Diagnostics::new();
    let parsed = ParsedDatabase::new(&diags, &source);

    // TODO don't require top module?
    let _ = compile(
        &diags,
        &source,
        &parsed,
        ElaborationSet::AsMuchAsPossible,
        &mut NoPrintHandler,
        &|| false,
        NON_ZERO_USIZE_ONE,
    );
}
