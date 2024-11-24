use crate::data::diagnostic::ErrorGuaranteed;
use crate::new::types::ClosedIntRange;
use crate::new_index_type;
use crate::syntax::ast::{PortDirection, SyncDomain};
use crate::util::arena::Arena;
use num_bigint::BigUint;

// Variant of `Type` that can only represent types that are valid in hardware.
#[derive(Debug)]
pub enum IrType {
    Clock,
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
    pub name: String,
    pub ty: IrType,
}

#[derive(Debug)]
pub struct IrWireInfo {
    pub name: String,
    pub ty: IrType,
}

#[derive(Debug)]
pub struct IrVariableInfo {
    pub name: String,
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
    pub locals: Arena<IrVariable, IrVariableInfo>,
    pub body: IrBlock,
}

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

#[derive(Debug)]
pub enum IrExpression {
    // TODO
}