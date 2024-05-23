use lalrpop_util::lalrpop_mod;

pub mod parse;
pub mod util;

lalrpop_mod!(pub grammar);

// #[cfg(test)]
// mod tests {
//     use super::grammar;
// 
//     #[test]
//     fn basic() {
//         println!("{:?}", grammar::ExprParser::new().parse("2*3+2"));
//         println!("{:?}", grammar::ExprParser::new().parse("2*(3+2)"));
//     }
// }