use std::collections::HashSet;
use std::vec;

use crate::front::types::{ClosedIncRange, HardwareType, Type};
use crate::front::value::CompileValue;
use crate::new_index_type;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, ResetKind, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::int::IntRepresentation;
use unwrap_match::unwrap_match;

// TODO add an "optimization" pass that does some basic stuff like:
//   * inline variables that are always equal to some other variable
//   * skip empty blocks, skip useless ifs, ...
//   That would allow the frontend to be freely generate redundant code,
//     while still getting clean output RTL.
// TODO dropping this type takes a long time (maybe due to the web of vecs caused by blocks/statements/...?)
#[derive(Debug)]
pub struct IrDatabase {
    pub modules: IrModules,
    pub top_module: IrModule,
}

pub type IrModules = Arena<IrModule, IrModuleInfo>;

// TODO check for circular instantiations
pub fn ir_modules_topological_sort(modules: &IrModules, top: IrModule) -> Vec<IrModule> {
    let mut result = vec![];
    // using a hashmap here is fine, we don't rely on its ordering anywhere
    let mut seen = HashSet::new();
    let mut todo = vec![top];

    while let Some(module) = todo.pop() {
        if seen.contains(&module) {
            continue;
        }

        seen.insert(module);
        result.push(module);

        for child in modules[module].children.iter().rev() {
            match child {
                IrModuleChild::ModuleInstance(inst) => {
                    todo.push(inst.module);
                }
                IrModuleChild::ClockedProcess(_) | IrModuleChild::CombinatorialProcess(_) => {}
            }
        }
    }

    result.reverse();
    result
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
    pub large: IrLargeArena,

    pub children: Vec<IrModuleChild>,

    pub debug_info_id: MaybeIdentifier,
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

// TODO change the execution model and block representation:
//  * processes must explicitly request signals to read and report signals they drive
//  * these read/write things correspond to IR variables that are read at the start, and always fully written at the end
//  careful: would that style of codegen break automatic clock gating?

/// The execution/memory model is:
/// * all writes are immediately visible to later reads in the current block
/// * writes only become visible to other blocks once all blocks (recursively) triggered by the current event
///   have fully finished running
///
/// If a local is read without being written to, the resulting value is undefined.
#[derive(Debug)]
pub struct IrClockedProcess {
    // TODO rename to variables
    pub locals: IrVariables,
    pub clock_signal: Spanned<IrExpression>,
    pub clock_block: IrBlock,
    pub async_reset_signal_and_block: Option<(Spanned<IrExpression>, IrBlock)>,
}

pub struct IrResetInfo {
    pub kind: ResetKind,
    pub signal: Spanned<IrExpression>,
    pub block: IrBlock,
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
    pub port_connections: Vec<Spanned<IrPortConnection>>,
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
    PrintLn(String),
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
    pub steps: Vec<IrTargetStep>,
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
pub enum IrTargetStep {
    ArrayIndex(IrExpression),
    ArraySlice(IrExpression, BigUint),
}

// TODO find better name than "large"
new_index_type!(pub IrExpressionLargeIndex);
pub type IrLargeArena = Arena<IrExpressionLargeIndex, IrExpressionLarge>;

// TODO consider not nesting these, but forcing a pass through a local variable for compound expressions
//   that should simplify the backends, at the cost of more verbose backend codegen
// TODO _dropping_ IrExpressions is taking a long time, they should really be stored in an arena per module instead
#[derive(Debug, Clone)]
pub enum IrExpression {
    // constants
    Bool(bool),
    Int(BigInt),

    // "signals"
    Port(IrPort),
    Wire(IrWire),
    Register(IrRegister),
    Variable(IrVariable),

    // expressions that need allocations,
    //   we don't want a web of boxes which is slow to construct and drop
    Large(IrExpressionLargeIndex),
}

#[derive(Debug)]
pub enum IrExpressionLarge {
    // actual expressions
    BoolNot(IrExpression),
    BoolBinary(IrBoolBinaryOp, IrExpression, IrExpression),
    IntArithmetic(IrIntArithmeticOp, ClosedIncRange<BigInt>, IrExpression, IrExpression),
    IntCompare(IrIntCompareOp, IrExpression, IrExpression),

    // concat
    TupleLiteral(Vec<IrExpression>),
    // TODO remove spread operator from this, replace with concat operator?
    ArrayLiteral(IrType, BigUint, Vec<IrArrayLiteralElement>),

    // slice
    ArrayIndex {
        base: IrExpression,
        index: IrExpression,
    },
    ArraySlice {
        base: IrExpression,
        start: IrExpression,
        len: BigUint,
    },

    // casting
    // TODO these first two should be any type, not just int
    // to-bits can never fail
    IntToBits(ClosedIncRange<BigInt>, IrExpression),
    // from-bits can fail (eg. if the resulting value if out of range), if it fails the resulting value is undefined
    IntFromBits(ClosedIncRange<BigInt>, IrExpression),

    // expand can never fail, this is just a re-encoding
    ExpandIntRange(ClosedIncRange<BigInt>, IrExpression),
    // constrain can fail, if it fails the resulting value is undefined
    ConstrainIntRange(ClosedIncRange<BigInt>, IrExpression),
}

// TODO move to separate utils module?
impl IrLargeArena {
    pub fn push_expr(&mut self, info: IrExpressionLarge) -> IrExpression {
        IrExpression::Large(self.push(info))
    }
}

#[derive(Debug, Clone)]
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
            IrType::Bool => BigUint::ONE,
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

            &IrExpression::Port(port) => module.ports[port].ty.clone(),
            &IrExpression::Wire(wire) => module.wires[wire].ty.clone(),
            &IrExpression::Register(reg) => module.registers[reg].ty.clone(),
            &IrExpression::Variable(var) => locals[var].ty.clone(),

            &IrExpression::Large(expr) => {
                match &module.large[expr] {
                    IrExpressionLarge::BoolNot(_) => IrType::Bool,
                    IrExpressionLarge::BoolBinary(_, left, _) => left.ty(module, locals),
                    IrExpressionLarge::IntArithmetic(_, ty, _, _) => IrType::Int(ty.clone()),
                    IrExpressionLarge::IntCompare(_, _, _) => IrType::Bool,

                    IrExpressionLarge::TupleLiteral(v) => {
                        IrType::Tuple(v.iter().map(|x| x.ty(module, locals)).collect())
                    }
                    IrExpressionLarge::ArrayLiteral(ty_inner, len, _values) => {
                        IrType::Array(Box::new(ty_inner.clone()), len.clone())
                    }

                    // TODO store resulting type in expression instead?
                    IrExpressionLarge::ArrayIndex { base, .. } => {
                        unwrap_match!(base.ty(module, locals), IrType::Array(inner, _) => *inner)
                    }
                    IrExpressionLarge::ArraySlice { base, start: _, len } => {
                        let inner = unwrap_match!(base.ty(module, locals), IrType::Array(inner, _) => inner);
                        IrType::Array(inner, len.clone())
                    }

                    IrExpressionLarge::IntToBits(ty, _) => {
                        IrType::Array(Box::new(IrType::Bool), IntRepresentation::for_range(ty).width)
                    }
                    IrExpressionLarge::IntFromBits(ty, _) => IrType::Int(ty.clone()),
                    IrExpressionLarge::ExpandIntRange(ty, _) => IrType::Int(ty.clone()),
                    IrExpressionLarge::ConstrainIntRange(ty, _) => IrType::Int(ty.clone()),
                }
            }
        }
    }

    pub fn to_diagnostic_string(&self, module: &IrModuleInfo) -> String {
        match self {
            IrExpression::Bool(x) => x.to_string(),
            IrExpression::Int(x) => x.to_string(),

            &IrExpression::Port(x) => module.ports[x].debug_info_id.string.clone(),
            &IrExpression::Wire(x) => module.wires[x].debug_info_id.string().to_owned(),
            &IrExpression::Register(x) => module.registers[x].debug_info_id.string().to_owned(),
            // TODO support printing variables with their real names if in a context where they exist
            &IrExpression::Variable(_) => "_variable".to_owned(),

            &IrExpression::Large(expr) => match &module.large[expr] {
                IrExpressionLarge::BoolNot(x) => format!("!({})", x.to_diagnostic_string(module)),
                IrExpressionLarge::BoolBinary(op, left, right) => {
                    let op_str = match op {
                        IrBoolBinaryOp::And => "&&",
                        IrBoolBinaryOp::Or => "||",
                        IrBoolBinaryOp::Xor => "^",
                    };
                    format!(
                        "({} {} {})",
                        left.to_diagnostic_string(module),
                        op_str,
                        right.to_diagnostic_string(module)
                    )
                }
                IrExpressionLarge::IntArithmetic(op, ty, left, right) => {
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
                        left.to_diagnostic_string(module),
                        op_str,
                        right.to_diagnostic_string(module)
                    )
                }
                IrExpressionLarge::IntCompare(op, left, right) => {
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
                        left.to_diagnostic_string(module),
                        op_str,
                        right.to_diagnostic_string(module)
                    )
                }

                IrExpressionLarge::TupleLiteral(v) => {
                    let v_str = v
                        .iter()
                        .map(|x| x.to_diagnostic_string(module))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({})", v_str)
                }
                IrExpressionLarge::ArrayLiteral(ty, len, v) => {
                    let ty_str = ty.to_diagnostic_string();
                    let v_str = v
                        .iter()
                        .map(|x| match x {
                            IrArrayLiteralElement::Single(value) => value.to_diagnostic_string(module),
                            IrArrayLiteralElement::Spread(value) => format!("*{}", value.to_diagnostic_string(module)),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{ty_str} {len}; {v_str}]")
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    format!(
                        "({}[{}])",
                        base.to_diagnostic_string(module),
                        index.to_diagnostic_string(module)
                    )
                }
                IrExpressionLarge::ArraySlice { base, start, len } => {
                    format!(
                        "({}[{}..+{}])",
                        base.to_diagnostic_string(module),
                        start.to_diagnostic_string(module),
                        len
                    )
                }
                IrExpressionLarge::IntToBits(ty, x) => {
                    format!("int_to_bits({}, {})", ty, x.to_diagnostic_string(module))
                }
                IrExpressionLarge::IntFromBits(ty, x) => {
                    format!("int_from_bits({}, {})", ty, x.to_diagnostic_string(module))
                }
                IrExpressionLarge::ExpandIntRange(ty, x) => {
                    format!("expand_int_range({}, {})", ty, x.to_diagnostic_string(module))
                }
                IrExpressionLarge::ConstrainIntRange(ty, x) => {
                    format!("constrain_int_range({}, {})", ty, x.to_diagnostic_string(module))
                }
            },
        }
    }

    pub fn for_each_expression_operand(&self, large: &IrLargeArena, f: &mut impl FnMut(&IrExpression)) {
        match self {
            IrExpression::Bool(_)
            | IrExpression::Int(_)
            | IrExpression::Port(_)
            | IrExpression::Wire(_)
            | IrExpression::Register(_)
            | IrExpression::Variable(_) => {}

            &IrExpression::Large(expr) => match &large[expr] {
                IrExpressionLarge::BoolNot(x) => f(x),
                IrExpressionLarge::BoolBinary(_op, left, right) => {
                    f(left);
                    f(right);
                }
                IrExpressionLarge::IntArithmetic(_op, _ty, left, right) => {
                    f(left);
                    f(right);
                }
                IrExpressionLarge::IntCompare(_op, left, right) => {
                    f(left);
                    f(right);
                }
                IrExpressionLarge::TupleLiteral(x) => x.iter().for_each(f),
                IrExpressionLarge::ArrayLiteral(_ty, _len, x) => x.iter().for_each(|a| match a {
                    IrArrayLiteralElement::Single(v) | IrArrayLiteralElement::Spread(v) => f(v),
                }),
                IrExpressionLarge::ArrayIndex { base, index } => {
                    f(base);
                    f(index);
                }
                IrExpressionLarge::ArraySlice { base, start, len: _ } => {
                    f(base);
                    f(start);
                }
                IrExpressionLarge::IntToBits(_ty, x) | IrExpressionLarge::IntFromBits(_ty, x) => f(x),
                IrExpressionLarge::ExpandIntRange(_ty, x) => f(x),
                IrExpressionLarge::ConstrainIntRange(_ty, x) => f(x),
            },
        }
    }

    pub fn contains_variable(&self, large: &IrLargeArena) -> bool {
        if let IrExpression::Variable(_) = self {
            return true;
        }

        // TODO short circuit
        let mut any = false;
        self.for_each_expression_operand(large, &mut |expr| {
            any |= expr.contains_variable(large);
        });
        any
    }
}
