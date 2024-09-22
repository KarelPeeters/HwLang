use crate::data::diagnostic::ErrorGuaranteed;

pub struct LoweredDatabase {
    pub top_module_name: Result<String, ErrorGuaranteed>,
    pub verilog_source: String,
}
