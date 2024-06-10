use std::io::Write;

pub struct CompileSet {

}

pub struct CheckedCompileSet {

}

impl CompileSet {
    pub fn add_package(&mut self, path: Vec<String>, source: String) {
        todo!()
    }

    pub fn add_external_vhdl(&mut self, library: String, source: String) {
        todo!()
    }

    pub fn add_external_verilog(&mut self, source: String) {
        todo!()
    }

    pub fn check() -> CheckedCompileSet {
        todo!()
    }
}

impl CheckedCompileSet {
    pub fn export(&self, f: impl Write) -> std::fmt::Result {
        todo!()
    }
}
