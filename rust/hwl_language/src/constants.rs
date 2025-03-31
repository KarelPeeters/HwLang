pub const LANGUAGE_NAME_FULL: &str = "HWLang";
pub const LANGUAGE_FILE_EXTENSION: &str = "kh";

pub const LSP_SERVER_NAME: &str = "HWLang-LSP";

// TODO maybe we can reduce this by now, module elaboration does not count towards the stack any more
//   it might also not matter, maybe every platform pre-commits stack space by now
pub const THREAD_STACK_SIZE: usize = 1024 * 1024 * 1024;
