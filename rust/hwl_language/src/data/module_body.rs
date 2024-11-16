use crate::data::compiled::{Item, ModulePort, Register, Wire};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::module::Driver;
use crate::front::value::DomainSignal;
use crate::syntax::ast;
use crate::syntax::ast::{Expression, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use indexmap::IndexMap;

// TODO include body comments for eg. the values that values were resolved to
#[derive(Debug, Clone)]
pub struct LowerModule<'a> {
    pub statements: Vec<LowerModuleStatement<'a>>,
    pub regs: Vec<Register>,
    pub wires: Vec<(Wire, Option<&'a Expression>)>,
    pub output_port_driver: IndexMap<ModulePort, Driver>,
}

#[derive(Debug, Clone)]
pub enum LowerModuleStatement<'a> {
    Instance(LowerModuleInstance<'a>),
    Combinatorial(LowerBlockCombinatorial<'a>),
    Clocked(LowerBlockClocked<'a>),
    Err(ErrorGuaranteed),
}

#[derive(Debug, Clone)]
pub struct LowerBlockCombinatorial<'a> {
    pub span: Span,
    // TODO include necessary temporary variables and other metadata
    pub block: LowerBlock<'a>,
}

#[derive(Debug, Clone)]
pub struct LowerBlockClocked<'a> {
    pub span: Span,
    pub domain: SyncDomain<DomainSignal>,
    pub on_reset: LowerBlock<'a>,
    pub on_clock: LowerBlock<'a>,
}

#[derive(Debug, Clone)]
pub struct LowerBlock<'a> {
    pub statements: Vec<Spanned<LowerStatement<'a>>>,
}

#[derive(Debug, Clone)]
pub enum LowerStatement<'a> {
    Error(ErrorGuaranteed),

    Block(LowerBlock<'a>),
    Expression(&'a ast::Expression),

    Assignment { target: &'a ast::Expression, value: &'a ast::Expression },

    // TODO we don't support any expressions with side effects (yet), does this make sense?
    If(LowerIfStatement<'a>),
    //TODO
    For,
    While,
    Return(Option<&'a ast::Expression>),
}

#[derive(Debug, Clone)]
pub struct LowerIfStatement<'a> {
    pub condition: &'a ast::Expression,
    pub then_block: LowerBlock<'a>,
    pub else_block: Option<LowerBlock<'a>>,
}

#[derive(Debug, Clone)]
pub struct LowerModuleInstance<'a> {
    pub module: Item,
    pub name: Option<String>,
    pub generic_arguments: Option<Vec<&'a ast::Expression>>,
    pub port_connections: Vec<&'a ast::Expression>,
}
