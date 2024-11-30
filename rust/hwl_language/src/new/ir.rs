use crate::data::diagnostic::ErrorGuaranteed;
use crate::new::types::{ClosedIntRange, Type};
use crate::new_index_type;
use crate::syntax::ast::{PortDirection, SyncDomain};
use crate::util::arena::Arena;
use crate::util::int::IntRepresentation;
use num_bigint::{BigInt, BigUint};
use num_traits::One;

// Variant of `Type` that can only represent types that are valid in hardware.
#[derive(Debug)]
pub enum IrType {
    Bool,
    Int(ClosedIntRange),
    Array(Box<IrType>, BigUint),
}

#[derive(Debug)]
pub struct IrDesign {
    pub top_module: Result<IrModule, ErrorGuaranteed>,
    pub modules: Arena<IrModule, IrModuleInfo>,
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

    pub processes: Vec<IrProcess>,
}

#[derive(Debug)]
pub struct IrPortInfo {
    pub name: String,
    pub direction: PortDirection,
    pub ty: IrType,
}

#[derive(Debug)]
pub struct IrRegisterInfo {
    pub debug_name: Option<String>,
    pub ty: IrType,
}

#[derive(Debug)]
pub struct IrWireInfo {
    pub debug_name: Option<String>,
    pub ty: IrType,
}

#[derive(Debug)]
pub struct IrVariableInfo {
    pub debug_name: Option<String>,
    pub ty: IrType,
}

#[derive(Debug)]
pub enum IrProcess {
    Clocked(SyncDomain<IrExpression>, IrProcessBody),
    Combinatorial(IrProcessBody),
}

/// The execution/memory model is:
/// * all writes are immediately visible to future reads in the current block
/// * writes only become visible to other blocks once all blocks (recursively) triggered by the current event
/// have fully finished running
///
/// If a local is read without being written to, the resulting value is undefined.
#[derive(Debug)]
pub struct IrProcessBody {
    pub locals: IrLocals,
    pub block: IrBlock,
}

pub type IrPorts = Arena<IrPort, IrPortInfo>;
pub type IrWires = Arena<IrWire, IrWireInfo>;
pub type IrRegisters = Arena<IrRegister, IrRegisterInfo>;
pub type IrLocals = Arena<IrVariable, IrVariableInfo>;

#[derive(Debug)]
pub enum IrSignal {
    Port(IrPort),
    Register(IrRegister),
    Wire(IrWire),
    // TODO are variables not just wires?
    Variable(IrVariable),
}

#[derive(Debug)]
pub struct IrBlock {
    pub statements: Vec<IrStatement>,
}

#[derive(Debug)]
pub enum IrStatement {
    // Assign(IrVariable, IrExpression),
    // If(IrExpression, IrBlock, IrBlock),
    // While(IrExpression, IrBlock),
    // Break,
    // Continue,
    // Return(IrExpression),
}

#[derive(Debug, Clone)]
pub enum IrExpression {
    // constants
    Bool(bool),
    Int(BigInt),
    Array(Vec<IrExpression>),
    // "signals"
    Port(IrPort),
    Wire(IrWire),
    Register(IrRegister),
    Variable(IrVariable),
    // actual expressions
    BoolNot(Box<IrExpression>),
}

impl IrType {
    pub fn as_type(&self) -> Type {
        match self {
            IrType::Bool => Type::Bool,
            IrType::Int(range) => Type::Int(range.clone().into_range()),
            IrType::Array(inner, len) => Type::Array(Box::new(inner.as_type()), len.clone()),
        }
    }

    pub fn bit_width(&self) -> BigUint {
        match self {
            IrType::Bool => BigUint::one(),
            IrType::Int(range) => {
                let ClosedIntRange { start_inc, end_inc } = range;
                IntRepresentation::for_range(start_inc.clone()..=end_inc.clone()).bits
            }
            IrType::Array(inner, len) => inner.bit_width() * len,
        }
    }
}