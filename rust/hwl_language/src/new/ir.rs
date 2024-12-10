use crate::data::diagnostic::ErrorGuaranteed;
use crate::new::types::{ClosedIntRange, HardwareType, Type};
use crate::new::value::CompileValue;
use crate::new_index_type;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::util::arena::Arena;
use crate::util::int::IntRepresentation;
use num_bigint::{BigInt, BigUint};
use num_traits::One;

/// Variant of `Type` that can only represent types that are valid in hardware.
#[derive(Debug)]
pub enum IrType {
    Bool,
    Int(ClosedIntRange),
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

    pub processes: Vec<Spanned<IrProcess>>,

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
pub enum IrProcess {
    Clocked(IrClockedProcess),
    Combinatorial(IrCombinatorialProcess),
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

pub type IrPorts = Arena<IrPort, IrPortInfo>;
pub type IrWires = Arena<IrWire, IrWireInfo>;
pub type IrRegisters = Arena<IrRegister, IrRegisterInfo>;
pub type IrVariables = Arena<IrVariable, IrVariableInfo>;

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
    pub statements: Vec<Spanned<IrStatement>>,
}

#[derive(Debug)]
pub enum IrStatement {
    Assign(IrAssignmentTarget, IrExpression),
    // If(IrExpression, IrBlock, IrBlock),
    // While(IrExpression, IrBlock),
    // Break,
    // Continue,
    // Return(IrExpression),
}

#[derive(Debug)]
pub enum IrAssignmentTarget {
    Port(IrPort),
    Register(IrRegister),
    Wire(IrWire),
    Variable(IrVariable),
}

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
    // actual expressions
    BoolNot(Box<IrExpression>),
    Array(Vec<IrExpression>),
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
            IrType::Int(range) => IntRepresentation::for_range(range).width,
            IrType::Array(inner, len) => inner.bit_width() * len,
        }
    }
}

impl IrExpression {
    pub fn to_diagnostic_string(&self, m: &IrModuleInfo) -> String {
        match self {
            IrExpression::BoolNot(x) =>
                format!("!({})", x.to_diagnostic_string(m)),

            IrExpression::Bool(x) => x.to_string(),
            IrExpression::Int(x) => x.to_string(),
            IrExpression::Array(x) => {
                let inner = x.iter().map(|x| x.to_diagnostic_string(m)).collect::<Vec<_>>().join(", ");
                format!("[{inner}]")
            }
            &IrExpression::Port(x) => m.ports[x].debug_info_id.string.clone(),
            &IrExpression::Wire(x) => m.wires[x].debug_info_id.string().to_owned(),
            &IrExpression::Register(x) => m.registers[x].debug_info_id.string().to_owned(),

            // TODO support printing variables with their real names if in a context where they exist
            &IrExpression::Variable(x) => "_variable".to_owned(),
        }
    }
}