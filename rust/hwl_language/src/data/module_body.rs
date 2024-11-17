// use crate::data::compiled::{Item, ModulePort, Register, Wire};
// use crate::data::diagnostic::ErrorGuaranteed;
// use crate::front::module::Driver;
// use crate::front::types::{GenericArguments, PortConnections};
// use crate::front::values::Value;
// use crate::syntax::ast::{Spanned, SyncDomain};
// use crate::syntax::pos::Span;
// use indexmap::IndexMap;
// 
// // TODO include body comments for eg. the values that values were resolved to
// #[derive(Debug, Clone)]
// pub struct ModuleChecked {
//     pub statements: Vec<ModuleStatement>,
//     pub regs: Vec<Register>,
//     pub wires: Vec<(Wire, Option<Value>)>,
//     pub output_port_driver: IndexMap<ModulePort, Driver>,
// }
// 
// #[derive(Debug, Clone)]
// pub enum ModuleStatement {
//     Instance(ModuleInstance),
//     Combinatorial(ModuleBlockCombinatorial),
//     Clocked(ModuleBlockClocked),
//     Err(ErrorGuaranteed),
// }
// 
// #[derive(Debug, Clone)]
// pub struct ModuleBlockCombinatorial {
//     pub span: Span,
//     // TODO include necessary temporary variables and other metadata
//     pub block: LowerBlock,
// }
// 
// #[derive(Debug, Clone)]
// pub struct ModuleBlockClocked {
//     pub span: Span,
//     pub domain: SyncDomain<Value>,
//     pub on_reset: LowerBlock,
//     pub on_clock: LowerBlock,
// }
// 
// #[derive(Debug, Clone)]
// pub struct LowerBlock {
//     pub statements: Vec<Spanned<LowerStatement>>,
// }
// 
// #[derive(Debug, Clone)]
// pub enum LowerStatement {
//     Error(ErrorGuaranteed),
// 
//     Block(LowerBlock),
//     Expression(Spanned<Value>),
// 
//     Assignment { target: Spanned<Value>, value: Spanned<Value> },
// 
//     // TODO we don't support any expressions with side effects (yet), does this make sense?
//     If(LowerIfStatement),
//     //TODO
//     For,
//     While,
//     Return(Option<Value>),
// }
// 
// #[derive(Debug, Clone)]
// pub struct LowerIfStatement {
//     pub condition: Spanned<Value>,
//     pub then_block: LowerBlock,
//     pub else_block: Option<LowerBlock>,
// }
// 
// #[derive(Debug, Clone)]
// pub struct ModuleInstance {
//     pub module: Item,
//     pub name: Option<String>,
//     pub generic_arguments: Option<GenericArguments>,
//     pub port_connections: PortConnections,
// }
