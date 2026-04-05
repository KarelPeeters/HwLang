#![no_main]

use hwl_common::diagnostic::Diagnostics;
use hwl_common::source::SourceDatabase;
use hwl_common::util::NON_ZERO_USIZE_ONE;
use hwl_language::front::compile::{CompileFixed, CompileRefs, CompileSettings, CompileShared, QueueItems};
use hwl_language::front::print::IgnorePrintHandler;
use hwl_language::syntax::hierarchy::SourceHierarchy;
use hwl_language::syntax::parsed::ParsedDatabase;
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

    let settings = CompileSettings { do_ir_cleanup: true };
    let fixed = CompileFixed {
        settings: &settings,
        source: &source,
        hierarchy: &hierarchy,
        parsed: &parsed,
    };

    let shared = CompileShared::new(&diags, fixed, QueueItems::All, NON_ZERO_USIZE_ONE);
    let refs = CompileRefs {
        diags: &diags,
        fixed,
        shared: &shared,
        print_handler: &IgnorePrintHandler,
        should_stop: &|| false,
    };
    refs.run_compile_loop(None);
}
