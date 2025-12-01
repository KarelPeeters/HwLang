#![no_main]

use hwl_language::front::compile::{ElaborationSet, compile};
use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::front::print::IgnorePrintHandler;
use hwl_language::syntax::hierarchy::SourceHierarchy;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::util::NON_ZERO_USIZE_ONE;
use libfuzzer_sys::fuzz_target;

// TODO we can be a lot smarter here, generate only valid asts, reuse identifiers from a limited pool, ...
fuzz_target!(|data: String| target(data));

fn target(data: String) {
    let diags = Diagnostics::new();
    let mut source = SourceDatabase::new();
    let mut hierarchy = SourceHierarchy::new();

    let file = source.add_file("dummy.kh".to_string(), data.to_string());
    let dummy_span = source.full_span(file);
    hierarchy
        .add_file(&diags, &source, dummy_span, &["dummy".to_owned()], file)
        .unwrap();

    let parsed = ParsedDatabase::new(&diags, &source, &hierarchy);

    // TODO don't require top module?
    let _ = compile(
        &diags,
        &source,
        &hierarchy,
        &parsed,
        ElaborationSet::AsMuchAsPossible,
        &mut IgnorePrintHandler,
        &|| false,
        NON_ZERO_USIZE_ONE,
        dummy_span,
    );
}
