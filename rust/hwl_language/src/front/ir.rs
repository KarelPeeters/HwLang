use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::types::{ClosedIncRange, HardwareType, Type};
use crate::front::value::CompileValue;
use crate::new_index_type;
use crate::syntax::ast::{Identifier, IfStatement, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::util::arena::Arena;
use crate::util::int::IntRepresentation;
use indexmap::IndexMap;
use num_bigint::{BigInt, BigUint};
use num_traits::One;

/// Variant of `Type` that can only represent types that are valid in hardware.
#[derive(Debug)]
pub enum IrType {
    Bool,
    Int(ClosedIncRange<BigInt>),
    Tuple(Vec<IrType>),
    Array(Box<IrType>, BigUint),
}

#[derive(Debug)]
pub struct IrDatabase {
    pub modules: Arena<IrModule, IrModuleInfo>,
    pub top_module: Result<IrModule, ErrorGuaranteed>,
}

new_index_type!(pub IrModule);
new_index_type!(pub IrPort);
new_index_type!(pub IrVariable);
new_index_type!(pub IrRegister);
new_index_type!(pub IrWire);

#[derive(Debug)]
pub struct IrModuleInfo {
    pub ports: Arena<IrPort, IrPortInfo>,
    pub registers: Arena<IrRegister, IrRegisterInfo>,
    pub wires: Arena<IrWire, IrWireInfo>,

    pub children: Vec<IrModuleChild>,

    pub debug_info_id: Identifier,
    pub debug_info_generic_args: Option<Vec<(Identifier, CompileValue)>>,
}

impl IrModuleInfo {
    pub fn instantiated_submodules(&self) -> Vec<IrModule> {
        // TODO fill in once instantiations are implemented
        vec![]
    }
}

#[derive(Debug)]
pub struct IrPortInfo {
    pub direction: PortDirection,
    pub ty: IrType,

    pub debug_info_id: Identifier,
    pub debug_info_ty: HardwareType,
    pub debug_info_domain: String,
}

#[derive(Debug)]
pub struct IrRegisterInfo {
    pub ty: IrType,

    pub debug_info_id: MaybeIdentifier,
    pub debug_info_ty: HardwareType,
    pub debug_info_domain: String,
}

#[derive(Debug)]
pub struct IrWireInfo {
    pub ty: IrType,

    pub debug_info_id: MaybeIdentifier,
    pub debug_info_ty: HardwareType,
    pub debug_info_domain: String,
}

#[derive(Debug)]
pub struct IrVariableInfo {
    pub ty: IrType,

    pub debug_info_id: MaybeIdentifier,
}

#[derive(Debug)]
pub enum IrModuleChild {
    ClockedProcess(IrClockedProcess),
    CombinatorialProcess(IrCombinatorialProcess),
    ModuleInstance(IrModuleInstance),
}

/// The execution/memory model is:
/// * all writes are immediately visible to later reads in the current block
/// * writes only become visible to other blocks once all blocks (recursively) triggered by the current event
///   have fully finished running
///
/// If a local is read without being written to, the resulting value is undefined.
#[derive(Debug)]
pub struct IrClockedProcess {
    pub domain: Spanned<SyncDomain<IrExpression>>,
    pub locals: IrVariables,
    pub on_clock: IrBlock,
    pub on_reset: IrBlock,
}

#[derive(Debug)]
pub struct IrCombinatorialProcess {
    pub locals: IrVariables,
    pub block: IrBlock,
}

#[derive(Debug)]
pub struct IrModuleInstance {
    pub name: Option<String>,
    pub module: IrModule,
    pub port_connections: IndexMap<IrPort, IrPortConnection>,
}

#[derive(Debug)]
pub enum IrPortConnection {
    Input(Spanned<IrExpression>),
    Output(Option<IrWireOrPort>),
}

#[derive(Debug, Copy, Clone)]
pub enum IrWireOrPort {
    Wire(IrWire),
    Port(IrPort),
}

pub type IrPorts = Arena<IrPort, IrPortInfo>;
pub type IrWires = Arena<IrWire, IrWireInfo>;
pub type IrRegisters = Arena<IrRegister, IrRegisterInfo>;
pub type IrVariables = Arena<IrVariable, IrVariableInfo>;

#[derive(Debug)]
pub struct IrBlock {
    pub statements: Vec<Spanned<IrStatement>>,
}

#[derive(Debug)]
pub enum IrStatement {
    Assign(IrAssignmentTarget, IrExpression),

    Block(IrBlock),
    If(IrIfStatement),
}

pub type IrIfStatement = IfStatement<IrExpression, IrBlock, Option<IrBlock>>;

#[derive(Debug)]
pub enum IrAssignmentTarget {
    Port(IrPort),
    Register(IrRegister),
    Wire(IrWire),
    Variable(IrVariable),
}

// TODO maybe IrExpression should always have explicit types, that would make lowering easier
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IrExpression {
    // constants
    Bool(bool),
    Int(BigInt),

    // "signals"
    Port(IrPort),
    Wire(IrWire),
    Register(IrRegister),
    Variable(IrVariable),

    // actual expressions
    BoolNot(Box<IrExpression>),
    BoolBinary(IrBoolBinaryOp, Box<IrExpression>, Box<IrExpression>),
    IntArithmetic(IrIntArithmeticOp, Box<IrExpression>, Box<IrExpression>),
    IntCompare(IrIntCompareOp, Box<IrExpression>, Box<IrExpression>),

    TupleLiteral(Vec<IrExpression>),
    ArrayLiteral(Vec<IrExpression>),
    ArrayIndex {
        base: Box<IrExpression>,
        index: Box<IrExpression>,
    },
    ArraySlice {
        base: Box<IrExpression>,
        range: ClosedIncRange<Box<IrExpression>>,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IrBoolBinaryOp {
    And,
    Or,
    Xor,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IrIntArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IrIntCompareOp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

impl IrBoolBinaryOp {
    pub fn eval(&self, left: bool, right: bool) -> bool {
        match self {
            IrBoolBinaryOp::And => left && right,
            IrBoolBinaryOp::Or => left || right,
            IrBoolBinaryOp::Xor => left ^ right,
        }
    }
}

impl IrIntCompareOp {
    pub fn eval(&self, left: &BigInt, right: &BigInt) -> bool {
        match self {
            IrIntCompareOp::Eq => left == right,
            IrIntCompareOp::Neq => left != right,
            IrIntCompareOp::Lt => left < right,
            IrIntCompareOp::Lte => left <= right,
            IrIntCompareOp::Gt => left > right,
            IrIntCompareOp::Gte => left >= right,
        }
    }
}

impl IrType {
    pub fn as_type(&self) -> Type {
        match self {
            IrType::Bool => Type::Bool,
            IrType::Int(range) => Type::Int(range.clone().into_range()),
            IrType::Tuple(inner) => Type::Tuple(inner.iter().map(IrType::as_type).collect()),
            IrType::Array(inner, len) => Type::Array(Box::new(inner.as_type()), len.clone()),
        }
    }

    pub fn bit_width(&self) -> BigUint {
        match self {
            IrType::Bool => BigUint::one(),
            IrType::Int(range) => IntRepresentation::for_range(range).width,
            IrType::Tuple(inner) => inner.iter().map(IrType::bit_width).sum(),
            IrType::Array(inner, len) => inner.bit_width() * len,
        }
    }
}

impl IrExpression {
    pub fn to_diagnostic_string(&self, m: &IrModuleInfo) -> String {
        match self {
            IrExpression::Bool(x) => x.to_string(),
            IrExpression::Int(x) => x.to_string(),

            &IrExpression::Port(x) => m.ports[x].debug_info_id.string.clone(),
            &IrExpression::Wire(x) => m.wires[x].debug_info_id.string().to_owned(),
            &IrExpression::Register(x) => m.registers[x].debug_info_id.string().to_owned(),
            // TODO support printing variables with their real names if in a context where they exist
            &IrExpression::Variable(_) => "_variable".to_owned(),

            IrExpression::BoolNot(x) => format!("!({})", x.to_diagnostic_string(m)),
            IrExpression::BoolBinary(op, left, right) => {
                let op_str = match op {
                    IrBoolBinaryOp::And => "&&",
                    IrBoolBinaryOp::Or => "||",
                    IrBoolBinaryOp::Xor => "^",
                };
                format!(
                    "({} {} {})",
                    left.to_diagnostic_string(m),
                    op_str,
                    right.to_diagnostic_string(m)
                )
            }
            IrExpression::IntArithmetic(op, left, right) => {
                let op_str = match op {
                    IrIntArithmeticOp::Add => "+",
                    IrIntArithmeticOp::Sub => "-",
                    IrIntArithmeticOp::Mul => "*",
                    IrIntArithmeticOp::Div => "/",
                    IrIntArithmeticOp::Mod => "%",
                    IrIntArithmeticOp::Pow => "**",
                };
                format!(
                    "({} {} {})",
                    left.to_diagnostic_string(m),
                    op_str,
                    right.to_diagnostic_string(m)
                )
            }
            IrExpression::IntCompare(op, left, right) => {
                let op_str = match op {
                    IrIntCompareOp::Eq => "==",
                    IrIntCompareOp::Neq => "!=",
                    IrIntCompareOp::Lt => "<",
                    IrIntCompareOp::Lte => "<=",
                    IrIntCompareOp::Gt => ">",
                    IrIntCompareOp::Gte => ">=",
                };

                format!(
                    "({} {} {})",
                    left.to_diagnostic_string(m),
                    op_str,
                    right.to_diagnostic_string(m)
                )
            }

            IrExpression::TupleLiteral(x) => {
                let inner = x
                    .iter()
                    .map(|x| x.to_diagnostic_string(m))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", inner)
            }
            IrExpression::ArrayLiteral(x) => {
                let inner = x
                    .iter()
                    .map(|x| x.to_diagnostic_string(m))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{inner}]")
            }
            IrExpression::ArrayIndex { base, index } => {
                format!("({}[{}])", base.to_diagnostic_string(m), index.to_diagnostic_string(m))
            }
            IrExpression::ArraySlice { base, range } => {
                let ClosedIncRange { start_inc, end_inc } = range;
                format!(
                    "({}[{}..{}])",
                    base.to_diagnostic_string(m),
                    start_inc.to_diagnostic_string(m),
                    end_inc.to_diagnostic_string(m)
                )
            }
        }
    }

    pub fn contains_variable(&self) -> bool {
        match self {
            IrExpression::Bool(_)
            | IrExpression::Int(_)
            | IrExpression::Port(_)
            | IrExpression::Wire(_)
            | IrExpression::Register(_) => false,
            IrExpression::Variable(_) => true,
            IrExpression::BoolNot(x) => x.contains_variable(),
            IrExpression::BoolBinary(_op, left, right) => left.contains_variable() || right.contains_variable(),
            IrExpression::IntArithmetic(_op, left, right) => left.contains_variable() || right.contains_variable(),
            IrExpression::IntCompare(_op, left, right) => left.contains_variable() || right.contains_variable(),
            IrExpression::TupleLiteral(x) => x.iter().any(|x| x.contains_variable()),
            IrExpression::ArrayLiteral(x) => x.iter().any(|x| x.contains_variable()),
            IrExpression::ArrayIndex { base, index } => base.contains_variable() || index.contains_variable(),
            IrExpression::ArraySlice {
                base,
                range: ClosedIncRange { start_inc, end_inc },
            } => base.contains_variable() || start_inc.contains_variable() || end_inc.contains_variable(),
        }
    }
}
