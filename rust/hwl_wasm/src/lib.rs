use hwl_language::data::diagnostic::Diagnostics;
use hwl_language::data::source::FilePath;
use hwl_language::data::source::SourceDatabase;
use hwl_language::front::driver::compile;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, wasmtest! (changed again)");
}

#[wasm_bindgen]
pub fn diag_count(src: String) -> u32 {
    let mut source = SourceDatabase::new();
    source.add_file(FilePath(vec!["top".to_owned()]), "top.kh".to_owned(), src).unwrap();

    let diag = Diagnostics::new();
    let _ = compile(&diag, &source);

    diag.finish().len() as u32
}
