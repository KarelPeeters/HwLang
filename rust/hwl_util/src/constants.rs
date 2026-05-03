#[macro_export]
macro_rules! hwl_manifest_file_name_macro {
    () => {
        "hwl.toml"
    };
}

pub const HWL_LANGUAGE_NAME: &str = "HWLang";
pub const HWL_LANGUAGE_NAME_SHORT: &str = "HWL";
pub const HWL_FILE_EXTENSION: &str = "kh";
pub const HWL_MANIFEST_FILE_NAME: &str = hwl_manifest_file_name_macro!();
pub const HWL_LSP_NAME: &str = "HWLang-LSP";
pub const HWL_VERSION: &str = env!("CARGO_PKG_VERSION");

// TODO make all of these configurable
// TODO maybe we can reduce this by now, module elaboration does not count towards the stack any more
//   it might also not matter, maybe every platform pre-commits stack space by now
pub const COMPILE_THREAD_STACK_SIZE: usize = 1024 * 1024 * 1024;

pub const STACK_OVERFLOW_STACK_LIMIT: usize = 1000;
pub const STACK_OVERFLOW_ERROR_ENTRIES_SHOWN: usize = 15;
