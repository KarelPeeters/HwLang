use lalrpop_util::lalrpop_mod;

// TODO move grammar to syntax dir
lalrpop_mod!(grammar);
pub mod syntax;

pub mod util;
