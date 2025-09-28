#[macro_export]
macro_rules! hwl_manifest_file_name_macro {
    () => {
        "hwl.toml"
    };
}

pub const HWL_LANGUAGE_NAME: &str = "HWLang";
pub const HWL_FILE_EXTENSION: &str = "kh";
pub const HWL_MANIFEST_FILE_NAME: &str = hwl_manifest_file_name_macro!();
pub const HWL_LSP_NAME: &str = "HWLang-LSP";
pub const HWL_VERSION: &str = env!("CARGO_PKG_VERSION");
