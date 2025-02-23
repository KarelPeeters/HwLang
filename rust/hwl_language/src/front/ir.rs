use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::types::{ClosedIncRange, HardwareType, Type};
use crate::front::value::CompileValue;
use crate::new_index_type;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::util::arena::Arena;
use crate::util::int::IntRepresentation;
use indexmap::IndexMap;
use num_bigint::{BigInt, BigUint};
use num_traits::One;
use unwrap_match::unwrap_match;

#[derive(Debug)]
pub struct IrDatabase {
    pub modules: Arena<IrModule, IrModuleInfo>,
    pub top_module: Result<IrModule, ErrorGuaranteed>,
}

/// Variant of [Type] that can only represent types that are valid in hardware.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IrType {
    Bool,
    Int(ClosedIncRange<BigInt>),
    Tuple(Vec<IrType>),
    Array(Box<IrType>, BigUint),
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
    // TODO rename to variables
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
    pub port_connections: IndexMap<IrPort, Spanned<IrPortConnection>>,
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

#[derive(Debug)]
pub struct IrIfStatement {
    pub condition: IrExpression,
    pub then_block: IrBlock,
    pub else_block: Option<IrBlock>,
}

#[derive(Debug)]
pub struct IrAssignmentTarget {
    pub base: IrAssignmentTargetBase,
    pub steps: Vec<IrAssignmentTargetStep>,
}

impl IrAssignmentTarget {
    pub fn simple(base: IrAssignmentTargetBase) -> Self {
        IrAssignmentTarget {
            base,
            steps: Vec::new(),
        }
    }

    pub fn wire(wire: IrWire) -> Self {
        IrAssignmentTarget::simple(IrAssignmentTargetBase::Wire(wire))
    }

    pub fn port(port: IrPort) -> Self {
        IrAssignmentTarget::simple(IrAssignmentTargetBase::Port(port))
    }

    pub fn register(reg: IrRegister) -> Self {
        IrAssignmentTarget::simple(IrAssignmentTargetBase::Register(reg))
    }

    pub fn variable(var: IrVariable) -> Self {
        IrAssignmentTarget::simple(IrAssignmentTargetBase::Variable(var))
    }
}

#[derive(Debug)]
pub enum IrAssignmentTargetBase {
    Port(IrPort),
    Register(IrRegister),
    Wire(IrWire),
    Variable(IrVariable),
}

// TODO re-use from frontend?
#[derive(Debug)]
pub enum IrAssignmentTargetStep {
    ArrayAccess {
        start: IrExpression,
        slice_len: Option<BigUint>,
    },
}

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
    IntArithmetic(
        IrIntArithmeticOp,
        ClosedIncRange<BigInt>,
        Box<IrExpression>,
        Box<IrExpression>,
    ),
    IntCompare(IrIntCompareOp, Box<IrExpression>, Box<IrExpression>),

    // concat
    TupleLiteral(Vec<IrExpression>),
    // TODO remove spread operator from this, replace with concat operator?
    ArrayLiteral(IrType, Vec<IrArrayLiteralElement>),

    // slice
    ArrayIndex {
        base: Box<IrExpression>,
        index: Box<IrExpression>,
    },
    ArraySlice {
        base: Box<IrExpression>,
        start: Box<IrExpression>,
        len: BigUint,
    },

    // casting
    IntToBits(ClosedIncRange<BigInt>, Box<IrExpression>),
    IntFromBits(ClosedIncRange<BigInt>, Box<IrExpression>),
    ExpandIntRange(ClosedIncRange<BigInt>, Box<IrExpression>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IrArrayLiteralElement {
    Single(IrExpression),
    Spread(IrExpression),
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

    pub fn to_diagnostic_string(&self) -> String {
        self.as_type().to_diagnostic_string()
    }
}

impl IrExpression {
    // TODO avoid clones
    pub fn ty(&self, module: &IrModuleInfo, locals: &IrVariables) -> IrType {
        match self {
            IrExpression::Bool(_) => IrType::Bool,
            IrExpression::Int(v) => IrType::Int(ClosedIncRange::single(v.clone())),

            IrExpression::Port(port) => module.ports[*port].ty.clone(),
            IrExpression::Wire(wire) => module.wires[*wire].ty.clone(),
            IrExpression::Register(reg) => module.registers[*reg].ty.clone(),
            IrExpression::Variable(var) => locals[*var].ty.clone(),

            IrExpression::BoolNot(_) => IrType::Bool,
            IrExpression::BoolBinary(_, left, _) => left.ty(module, locals),
            IrExpression::IntArithmetic(_, ty, _, _) => IrType::Int(ty.clone()),
            IrExpression::IntCompare(_, _, _) => IrType::Bool,

            IrExpression::TupleLiteral(v) => IrType::Tuple(v.iter().map(|x| x.ty(module, locals)).collect()),
            IrExpression::ArrayLiteral(ty, values) => IrType::Array(Box::new(ty.clone()), BigUint::from(values.len())),

            // TODO store resulting type in expression instead?
            IrExpression::ArrayIndex { base, .. } => {
                unwrap_match!(base.ty(module, locals), IrType::Array(inner, _) => *inner)
            }
            IrExpression::ArraySlice { base, start: _, len } => {
                let inner = unwrap_match!(base.ty(module, locals), IrType::Array(inner, _) => inner);
                IrType::Array(inner, len.clone())
            }

            IrExpression::IntToBits(ty, _) => {
                IrType::Array(Box::new(IrType::Bool), IntRepresentation::for_range(ty).width)
            }
            IrExpression::IntFromBits(ty, _) => IrType::Int(ty.clone()),
            IrExpression::ExpandIntRange(ty, _) => IrType::Int(ty.clone()),
        }
    }

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
            IrExpression::IntArithmetic(op, ty, left, right) => {
                let op_str = match op {
                    IrIntArithmeticOp::Add => "+",
                    IrIntArithmeticOp::Sub => "-",
                    IrIntArithmeticOp::Mul => "*",
                    IrIntArithmeticOp::Div => "/",
                    IrIntArithmeticOp::Mod => "%",
                    IrIntArithmeticOp::Pow => "**",
                };
                format!(
                    "({}; {} {} {})",
                    ty,
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

            IrExpression::TupleLiteral(v) => {
                let v_str = v
                    .iter()
                    .map(|x| x.to_diagnostic_string(m))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", v_str)
            }
            IrExpression::ArrayLiteral(ty, v) => {
                let ty_str = ty.to_diagnostic_string();
                let v_str = v
                    .iter()
                    .map(|x| match x {
                        IrArrayLiteralElement::Single(value) => value.to_diagnostic_string(m),
                        IrArrayLiteralElement::Spread(value) => format!("*{}", value.to_diagnostic_string(m)),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{ty_str}; {v_str}]")
            }
            IrExpression::ArrayIndex { base, index } => {
                format!("({}[{}])", base.to_diagnostic_string(m), index.to_diagnostic_string(m))
            }
            IrExpression::ArraySlice { base, start, len } => {
                format!(
                    "({}[{}..+{}])",
                    base.to_diagnostic_string(m),
                    start.to_diagnostic_string(m),
                    len
                )
            }
            IrExpression::IntToBits(ty, x) => format!("int_to_bits({}, {})", ty, x.to_diagnostic_string(m)),
            IrExpression::IntFromBits(ty, x) => format!("int_from_bits({}, {})", ty, x.to_diagnostic_string(m)),
            IrExpression::ExpandIntRange(ty, x) => format!("expand_int_range({}, {})", ty, x.to_diagnostic_string(m)),
        }
    }

    pub fn for_each_expression_operand(&self, f: &mut impl FnMut(&IrExpression)) {
        match self {
            IrExpression::Bool(_)
            | IrExpression::Int(_)
            | IrExpression::Port(_)
            | IrExpression::Wire(_)
            | IrExpression::Register(_)
            | IrExpression::Variable(_) => {}

            IrExpression::BoolNot(x) => f(x),
            IrExpression::BoolBinary(_op, left, right) => {
                f(left);
                f(right);
            }
            IrExpression::IntArithmetic(_op, _ty, left, right) => {
                f(left);
                f(right);
            }
            IrExpression::IntCompare(_op, left, right) => {
                f(left);
                f(right);
            }
            IrExpression::TupleLiteral(x) => x.iter().for_each(f),
            IrExpression::ArrayLiteral(_ty, x) => x.iter().for_each(|a| match a {
                IrArrayLiteralElement::Single(v) | IrArrayLiteralElement::Spread(v) => f(v),
            }),
            IrExpression::ArrayIndex { base, index } => {
                f(base);
                f(index);
            }
            IrExpression::ArraySlice { base, start, len: _ } => {
                f(base);
                f(start);
            }
            IrExpression::IntToBits(_ty, x) | IrExpression::IntFromBits(_ty, x) => f(x),
            IrExpression::ExpandIntRange(_ty, x) => f(x),
        }
    }

    pub fn contains_variable(&self) -> bool {
        if let IrExpression::Variable(_) = self {
            return true;
        }

        let mut any = false;
        self.for_each_expression_operand(&mut |expr| {
            any |= expr.contains_variable();
        });
        any
    }
}
