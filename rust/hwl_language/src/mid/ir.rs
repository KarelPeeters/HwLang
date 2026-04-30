//! Intermediate representation (IR) for hardware modules.
//!
//! The execution/memory model for processes is:
//! * all writes are immediately visible to later reads in the current block
//! * writes only become visible to other blocks once all blocks (recursively) triggered by the current event
//!   have fully finished running.
//!
//! If a local variable is read without being written to, the resulting value is undefined.

use crate::front::item::{ElaboratedEnum, ElaboratedStruct, HardwareChecked};
use crate::front::signal::Polarized;
use crate::front::types::HardwareType;
use crate::new_index_type;
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::syntax::pos::{Span, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::int::IntRepresentation;
use crate::util::range::{ClosedNonEmptyRange, ClosedRange};
use crate::util::range_multi::ClosedNonEmptyMultiRange;
use derive_more::From;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::sync::Arc;
use std::vec;
use unwrap_match::unwrap_match;

// TODO dropping this type takes a long time (maybe due to the web of vecs caused by blocks/statements/...?)
// TODO add some way to share Strings, especially for debug info there are lots of duplicates
#[derive(Debug, Clone)]
pub struct IrDatabase<M = IrModuleInfo> {
    pub modules: IrModules<M>,
    pub external_modules: IndexSet<String>,
}

pub type IrModules<M = IrModuleInfo> = Arena<IrModule, M>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum IrType {
    Bool,
    Int(ClosedNonEmptyRange<BigInt>),
    Array(Box<IrType>, BigUint),
    Tuple(Vec<IrType>),
    Struct(IrStructType),
    Enum(IrEnumType),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IrStructType {
    pub ty: HardwareChecked<ElaboratedStruct>,
    pub debug_info_name: String,
    pub fields: IndexMap<String, IrType>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IrEnumType {
    pub ty: HardwareChecked<ElaboratedEnum>,
    pub debug_info_name: String,
    // cannot be empty, that would cause the type to be non-representable
    pub variants: IndexMap<String, Option<IrType>>,
}

// TODO move down
impl IrStructType {
    pub fn size_bits(&self) -> BigUint {
        self.fields.values().map(|ty| ty.size_bits()).sum()
    }

    pub fn field_offset(&self, field: usize) -> BigUint {
        // TODO cache?
        self.fields[..field].iter().map(|(_, ty)| ty.size_bits()).sum()
    }
}

impl IrEnumType {
    pub fn size_bits(&self) -> BigUint {
        // TODO cache?
        self.tag_size_bits() + self.max_payload_size_bits()
    }

    pub fn tag_size_bits(&self) -> BigUint {
        // TODO cache?
        BigUint::from(IntRepresentation::for_range(self.tag_range().as_ref()).size_bits())
    }

    pub fn max_payload_size_bits(&self) -> BigUint {
        // TODO cache?
        self.variants
            .values()
            .filter_map(|payload| payload.as_ref())
            .map(|ty| ty.size_bits())
            .max()
            .unwrap_or(BigUint::ZERO)
    }

    pub fn tag_range(&self) -> ClosedNonEmptyRange<BigInt> {
        assert!(!self.variants.is_empty());
        ClosedNonEmptyRange {
            start: BigInt::ZERO,
            end: BigInt::from(self.variants.len()),
        }
    }
}

new_index_type!(pub IrModule);
new_index_type!(pub IrPort);
new_index_type!(pub IrWire);
new_index_type!(pub IrVariable);

#[derive(Debug, Clone)]
pub struct IrModuleInfo {
    pub ports: Arena<IrPort, IrPortInfo>,
    pub wires: Arena<IrWire, IrWireInfo>,
    pub large: IrLargeArena,

    // child names (if any) any not guaranteed to be unique
    pub children: Vec<Spanned<IrModuleChild>>,

    pub debug_info_location: String,
    pub debug_info_id: Spanned<Option<String>>,
    pub debug_info_generic_args: Option<Vec<(String, String)>>,
}

#[derive(Debug, Clone)]
pub struct IrPortInfo {
    pub name: String,
    pub direction: PortDirection,
    pub ty: IrType,

    pub debug_span: Span,
    pub debug_info_ty: Spanned<String>,
    pub debug_info_domain: String,
}

#[derive(Debug, Clone)]
pub struct IrWireInfo {
    pub ty: IrType,

    pub debug_info_id: Spanned<Option<String>>,
    pub debug_info_ty: String,
    pub debug_info_domain: String,
}

#[derive(Debug, Clone)]
pub struct IrVariableInfo {
    pub ty: IrType,

    pub debug_info_span: Span,
    pub debug_info_id: Option<String>,
}

#[derive(Debug, Clone)]
pub enum IrModuleChild {
    ClockedProcess(IrClockedProcess),
    CombinatorialProcess(IrCombinatorialProcess),
    ModuleInternalInstance(IrModuleInternalInstance),
    ModuleExternalInstance(IrModuleExternalInstance),
}

#[derive(Debug, Clone)]
pub struct IrClockedProcess {
    // TODO rename to variables
    pub locals: IrVariables,
    pub async_reset: Option<IrAsyncResetInfo>,
    pub clock_signal: Spanned<Polarized<IrSignal>>,
    pub clock_block: IrBlock,
}

#[derive(Debug, Clone)]
pub struct IrAsyncResetInfo {
    pub signal: Spanned<Polarized<IrSignal>>,
    // TODO make this a constant instead of an arbitrary expression, or maybe a "SimpleExpression"
    pub resets: Vec<Spanned<(IrSignal, IrExpression)>>,
}

#[derive(Debug, Clone)]
pub struct IrCombinatorialProcess {
    pub locals: IrVariables,
    pub block: IrBlock,
}

#[derive(Debug, Clone)]
pub struct IrModuleInternalInstance {
    pub name: Option<String>,
    pub module: IrModule,
    pub port_connections: Vec<Spanned<IrPortConnection>>,
}

// TODO ensure this works for zero-width ports
#[derive(Debug, Clone)]
pub struct IrModuleExternalInstance {
    pub name: Option<String>,
    pub module_name: String,
    pub generic_args: Option<Vec<(String, BigInt)>>,
    pub port_names: Vec<String>,
    pub port_connections: Vec<Spanned<IrPortConnection>>,
}

#[derive(Debug, Copy, Clone)]
pub enum IrPortConnection {
    Input(IrSignal),
    Output(Option<IrSignal>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, From)]
pub enum IrSignal {
    Port(IrPort),
    Wire(IrWire),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, From)]
pub enum IrSignalOrVariable {
    Signal(IrSignal),
    Variable(IrVariable),
}

pub type IrPorts = Arena<IrPort, IrPortInfo>;
pub type IrWires = Arena<IrWire, IrWireInfo>;
pub type IrVariables = Arena<IrVariable, IrVariableInfo>;

#[derive(Debug, Clone)]
pub struct IrBlock {
    pub statements: Vec<Spanned<IrStatement>>,
}

#[derive(Debug, Clone)]
pub enum IrStatement {
    Assign(IrAssignmentTarget, IrExpression),
    // TODO maybe we can remove this
    Block(IrBlock),
    If(IrIfStatement),
    For(IrForStatement),
    /// This does not automatically include a trailing newline.
    Print(IrString),
    AssertFailed,
}

pub type IrString = Vec<IrStringPiece>;
pub type IrStringPiece = StringPiece<String, IrStringSubstitution>;

#[derive(Debug, Clone)]
pub enum IrStringSubstitution {
    Integer(IrExpression, IrIntegerRadix),
}

#[derive(Debug, Copy, Clone)]
pub enum IrIntegerRadix {
    Binary,
    Decimal,
    Hexadecimal,
}

#[derive(Debug, Clone)]
pub struct IrIfStatement {
    pub condition: IrExpression,
    pub then_block: IrBlock,
    pub else_block: Option<IrBlock>,
}

#[derive(Debug, Clone)]
pub struct IrForStatement {
    // The index variable used to store the current value of the loop index.
    // Must have an integer type that can store the entire iteration range, plus the inclusive end of the range.
    // TODO relax this constraint, the backend can create an additional loop index variable if it really wants to
    pub index: IrVariable,
    pub range: ClosedRange<BigInt>,
    pub block: IrBlock,
}

#[derive(Debug, Clone)]
pub struct IrAssignmentTarget {
    pub base: IrSignalOrVariable,
    pub steps: Vec<IrTargetStep>,
}

impl IrAssignmentTarget {
    pub fn simple(base: IrSignalOrVariable) -> Self {
        IrAssignmentTarget { base, steps: vec![] }
    }
}

#[derive(Debug, Clone)]
pub enum IrTargetStep {
    ArrayIndex(IrExpression),
    ArraySlice { start: IrExpression, len: BigUint },
}

new_index_type!(pub IrExpressionLargeIndex);
pub type IrLargeArena = Arena<IrExpressionLargeIndex, IrExpressionLarge>;

// TODO consider not nesting these, but forcing a pass through a local variable for compound expressions
//   that should simplify the backends, at the cost of more verbose backend codegen
#[derive(Debug, Clone)]
pub enum IrExpression {
    Bool(bool),
    Int(BigInt),
    Signal(IrSignal),
    Variable(IrVariable),
    // expressions that need allocations,
    //   we don't want a web of boxes which is slow to construct and drop
    Large(IrExpressionLargeIndex),
}

#[derive(Debug, Clone)]
pub enum IrExpressionLarge {
    Undefined(IrType),

    // actual expressions
    BoolNot(IrExpression),
    BoolBinary(IrBoolBinaryOp, IrExpression, IrExpression),
    IntArithmetic(
        IrIntArithmeticOp,
        // The range of the resulting value.
        // This is a promise made from the frontend to the IR, if the frontend has additional information
        //   it can be tighter than the naive output range of the operation.
        ClosedNonEmptyRange<BigInt>,
        IrExpression,
        IrExpression,
    ),
    IntCompare(IrIntCompareOp, IrExpression, IrExpression),

    // concat
    // TODO always store these in intermediate variable to avoid large repeated expressions
    TupleLiteral(Vec<IrExpression>),
    ArrayLiteral(IrType, BigUint, Vec<IrArrayLiteralElement>),
    StructLiteral(IrStructType, Vec<IrExpression>),
    EnumLiteral(IrEnumType, usize, Option<IrExpression>),

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
    TupleIndex {
        base: IrExpression,
        index: usize,
    },
    StructField {
        base: IrExpression,
        field: usize,
    },
    EnumTag {
        base: IrExpression,
    },
    EnumPayload {
        base: IrExpression,
        variant: usize,
    },

    // casting
    // to-bits can never fail
    ToBits(IrType, IrExpression),
    // from-bits can fail (eg. for an int, if the resulting value if out of range),
    //   if so the result is undefined
    FromBits(IrType, IrExpression),
    // expand can never fail, this is just a re-encoding
    ExpandIntRange(ClosedNonEmptyRange<BigInt>, IrExpression),
    // constrain can fail, if it fails the resulting value is undefined
    ConstrainIntRange(ClosedNonEmptyRange<BigInt>, IrExpression),
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
    Shl,
    Shr,
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
    pub fn as_type_hw(&self) -> HardwareType {
        match self {
            IrType::Bool => HardwareType::Bool,
            IrType::Int(range) => HardwareType::Int(ClosedNonEmptyMultiRange::from(range.clone())),
            IrType::Array(inner, len) => HardwareType::Array(Arc::new(inner.as_type_hw()), len.clone()),
            IrType::Tuple(inner) => HardwareType::Tuple(Arc::new(inner.iter().map(IrType::as_type_hw).collect())),
            IrType::Struct(info) => HardwareType::Struct(info.ty),
            IrType::Enum(info) => HardwareType::Enum(info.ty),
        }
    }

    pub fn unwrap_int(self) -> ClosedNonEmptyRange<BigInt> {
        unwrap_match!(self, IrType::Int(range) => range)
    }

    pub fn unwrap_array(self) -> (IrType, BigUint) {
        unwrap_match!(self, IrType::Array(inner, len) => (*inner, len))
    }

    pub fn unwrap_tuple(self) -> Vec<IrType> {
        unwrap_match!(self, IrType::Tuple(elements) => elements)
    }

    pub fn unwrap_struct(self) -> IrStructType {
        unwrap_match!(self, IrType::Struct(info) => info)
    }

    pub fn unwrap_enum(self) -> IrEnumType {
        unwrap_match!(self, IrType::Enum(info) => info)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ValueAccess {
    Read,
    Write,
}

impl IrBlock {
    pub fn new_single(span: Span, stmt: IrStatement) -> Self {
        IrBlock {
            statements: vec![Spanned::new(span, stmt)],
        }
    }

    pub fn visit_values_accessed(&self, large: &IrLargeArena, f: &mut impl FnMut(IrSignalOrVariable, ValueAccess)) {
        let IrBlock { statements } = self;
        for stmt in statements {
            stmt.inner.visit_values_accessed(large, f);
        }
    }
}

impl IrStatement {
    pub fn visit_values_accessed(&self, large: &IrLargeArena, f: &mut impl FnMut(IrSignalOrVariable, ValueAccess)) {
        match self {
            IrStatement::Assign(target, expr) => {
                target.visit_values_accessed(large, f);
                expr.visit_values_accessed(large, f);
            }
            IrStatement::Block(block) => {
                block.visit_values_accessed(large, f);
            }
            IrStatement::If(if_stmt) => {
                let IrIfStatement {
                    condition,
                    then_block,
                    else_block,
                } = if_stmt;
                condition.visit_values_accessed(large, f);
                then_block.visit_values_accessed(large, f);
                if let Some(else_block) = else_block {
                    else_block.visit_values_accessed(large, f);
                }
            }
            IrStatement::For(for_stmt) => {
                let &IrForStatement {
                    index,
                    range: _,
                    ref block,
                } = for_stmt;
                f(IrSignalOrVariable::Variable(index), ValueAccess::Write);
                f(IrSignalOrVariable::Variable(index), ValueAccess::Read);
                block.visit_values_accessed(large, f);
            }
            IrStatement::Print(s) => visit_values_accessed_string(s, large, f),
            IrStatement::AssertFailed => {}
        }
    }
}

pub fn visit_values_accessed_string(
    pieces: &IrString,
    large: &IrLargeArena,
    f: &mut impl FnMut(IrSignalOrVariable, ValueAccess),
) {
    for p in pieces {
        match p {
            StringPiece::Literal(_) => {}
            StringPiece::Substitute(v) => {
                let v = match v {
                    IrStringSubstitution::Integer(v, _) => v,
                };
                v.visit_values_accessed(large, f);
            }
        }
    }
}

impl IrAssignmentTarget {
    pub fn visit_values_accessed(&self, large: &IrLargeArena, f: &mut impl FnMut(IrSignalOrVariable, ValueAccess)) {
        let &IrAssignmentTarget { base, ref steps } = self;

        f(base, ValueAccess::Write);
        for step in steps {
            match step {
                IrTargetStep::ArrayIndex(index) => index.visit_values_accessed(large, f),
                IrTargetStep::ArraySlice { start, len: _ } => start.visit_values_accessed(large, f),
            }
        }
    }
}

impl IrExpression {
    // TODO avoid clones
    pub fn ty(&self, module: &IrModuleInfo, locals: &IrVariables) -> IrType {
        match self {
            IrExpression::Bool(_) => IrType::Bool,
            IrExpression::Int(v) => IrType::Int(ClosedNonEmptyRange::single(v.clone())),

            &IrExpression::Signal(signal) => match signal {
                IrSignal::Port(port) => module.ports[port].ty.clone(),
                IrSignal::Wire(wire) => module.wires[wire].ty.clone(),
            },
            &IrExpression::Variable(var) => locals[var].ty.clone(),

            &IrExpression::Large(expr) => {
                match &module.large[expr] {
                    IrExpressionLarge::Undefined(ty) => ty.clone(),
                    IrExpressionLarge::BoolNot(_) => IrType::Bool,
                    IrExpressionLarge::BoolBinary(_, left, _) => left.ty(module, locals),
                    IrExpressionLarge::IntArithmetic(_, ty, _, _) => IrType::Int(ty.clone()),
                    IrExpressionLarge::IntCompare(_, _, _) => IrType::Bool,

                    IrExpressionLarge::ArrayLiteral(ty_inner, len, _values) => {
                        IrType::Array(Box::new(ty_inner.clone()), len.clone())
                    }
                    IrExpressionLarge::TupleLiteral(v) => {
                        IrType::Tuple(v.iter().map(|x| x.ty(module, locals)).collect())
                    }
                    IrExpressionLarge::StructLiteral(ty, _) => IrType::Struct(ty.clone()),
                    IrExpressionLarge::EnumLiteral(ty, _, _) => IrType::Enum(ty.clone()),

                    IrExpressionLarge::ArrayIndex { base, .. } => {
                        let (inner_ty, _) = base.ty(module, locals).unwrap_array();
                        inner_ty
                    }
                    IrExpressionLarge::ArraySlice { base, start: _, len } => {
                        let (inner_ty, _) = base.ty(module, locals).unwrap_array();
                        IrType::Array(Box::new(inner_ty), len.clone())
                    }
                    &IrExpressionLarge::TupleIndex { ref base, index } => {
                        let base_ty = base.ty(module, locals).unwrap_tuple();
                        base_ty[index].clone()
                    }
                    // TODO store resulting type in expression instead?
                    &IrExpressionLarge::StructField { ref base, field } => {
                        let base_ty = base.ty(module, locals).unwrap_struct();
                        base_ty.fields[field].clone()
                    }
                    IrExpressionLarge::EnumTag { base } => {
                        let base_ty = base.ty(module, locals).unwrap_enum();
                        IrType::Int(base_ty.tag_range())
                    }
                    &IrExpressionLarge::EnumPayload { ref base, variant } => {
                        let base_ty = base.ty(module, locals).unwrap_enum();
                        base_ty.variants[variant]
                            .as_ref()
                            .expect("cannot get payload of non-payload variant")
                            .clone()
                    }

                    IrExpressionLarge::ToBits(ty, _) => IrType::Array(Box::new(IrType::Bool), ty.size_bits()),
                    IrExpressionLarge::FromBits(ty, _) => ty.clone(),
                    IrExpressionLarge::ExpandIntRange(ty, _) => IrType::Int(ty.clone()),
                    IrExpressionLarge::ConstrainIntRange(ty, _) => IrType::Int(ty.clone()),
                }
            }
        }
    }

    pub fn visit_values_accessed(&self, large: &IrLargeArena, f: &mut impl FnMut(IrSignalOrVariable, ValueAccess)) {
        match *self {
            IrExpression::Signal(signal) => f(IrSignalOrVariable::Signal(signal), ValueAccess::Read),
            IrExpression::Variable(var) => f(IrSignalOrVariable::Variable(var), ValueAccess::Read),
            _ => {}
        }

        self.for_each_operand(large, &mut |op| op.visit_values_accessed(large, f));
    }

    pub fn for_each_operand(&self, large: &IrLargeArena, f: &mut impl FnMut(&IrExpression)) {
        match self {
            IrExpression::Bool(_) | IrExpression::Int(_) | IrExpression::Signal(_) | IrExpression::Variable(_) => {}

            &IrExpression::Large(expr) => match &large[expr] {
                IrExpressionLarge::Undefined(_ty) => {}
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
                IrExpressionLarge::StructLiteral(_ty, x) => x.iter().for_each(f),
                IrExpressionLarge::EnumLiteral(_ty, _, x) => {
                    if let Some(x) = x {
                        f(x);
                    }
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    f(base);
                    f(index);
                }
                IrExpressionLarge::ArraySlice { base, start, len: _ } => {
                    f(base);
                    f(start);
                }
                IrExpressionLarge::TupleIndex { base, index: _ } => f(base),
                IrExpressionLarge::StructField { base, field: _ } => {
                    f(base);
                }
                IrExpressionLarge::EnumTag { base } => {
                    f(base);
                }
                IrExpressionLarge::EnumPayload { base, variant: _ } => {
                    f(base);
                }
                IrExpressionLarge::ToBits(_ty, x) | IrExpressionLarge::FromBits(_ty, x) => f(x),
                IrExpressionLarge::ExpandIntRange(_ty, x) => f(x),
                IrExpressionLarge::ConstrainIntRange(_ty, x) => f(x),
            },
        }
    }

    /// This function does not mutate in-place,
    ///   that would be sketchy if a single expressions happens to be used in multiple places.
    pub fn map_recursive(
        &self,
        large: &mut IrLargeArena,
        f: &mut impl FnMut(&IrExpression) -> Option<IrExpression>,
    ) -> Option<IrExpression> {
        macro_rules! build_unary {
            (|$inner:ident| $build:expr) => {{
                match f($inner) {
                    None => None,
                    Some(inner_new) => {
                        let $inner = inner_new;
                        Some($build)
                    }
                }
            }};
        }
        macro_rules! build_binary {
            (|$left:ident, $right:ident| $build:expr) => {{
                let left = $left;
                let right = $right;
                let left_new = f(left);
                let right_new = f(right);
                if left_new.is_some() || right_new.is_some() {
                    let $left = left_new.unwrap_or_else(|| left.clone());
                    let $right = right_new.unwrap_or_else(|| right.clone());
                    Some($build)
                } else {
                    None
                }
            }};
        }

        fn build_nary(
            mut f: &mut impl FnMut(&IrExpression) -> Option<IrExpression>,
            elements: &[IrExpression],
        ) -> Option<Vec<IrExpression>> {
            let elements_new = elements.iter().map(&mut f).collect_vec();

            if elements_new.iter().any(|e| e.is_some()) {
                let elements_new = elements
                    .iter()
                    .zip(elements_new)
                    .map(|(old, new)| new.unwrap_or_else(|| old.clone()))
                    .collect_vec();
                Some(elements_new)
            } else {
                None
            }
        }

        // map operands
        let self_new = match self {
            IrExpression::Bool(_) | IrExpression::Int(_) | IrExpression::Signal(_) | IrExpression::Variable(_) => None,

            &IrExpression::Large(expr) => match &large[expr] {
                IrExpressionLarge::Undefined(_ty) => None,
                IrExpressionLarge::BoolNot(inner) => {
                    build_unary!(|inner| large.push_expr(IrExpressionLarge::BoolNot(inner)))
                }
                &IrExpressionLarge::BoolBinary(op, ref left, ref right) => {
                    build_binary!(|left, right| large.push_expr(IrExpressionLarge::BoolBinary(op, left, right)))
                }
                &IrExpressionLarge::IntArithmetic(op, ref ty, ref left, ref right) => {
                    build_binary!(|left, right| large.push_expr(IrExpressionLarge::IntArithmetic(
                        op,
                        ty.clone(),
                        left,
                        right
                    )))
                }
                &IrExpressionLarge::IntCompare(op, ref left, ref right) => {
                    build_binary!(|left, right| large.push_expr(IrExpressionLarge::IntCompare(op, left, right)))
                }
                IrExpressionLarge::TupleLiteral(elements) => {
                    build_nary(f, elements).map(|new| large.push_expr(IrExpressionLarge::TupleLiteral(new)))
                }
                IrExpressionLarge::ArrayLiteral(ty, len, elements) => {
                    let elements_new = elements
                        .iter()
                        .map(|e| match e {
                            IrArrayLiteralElement::Single(v) => f(v).map(IrArrayLiteralElement::Single),
                            IrArrayLiteralElement::Spread(v) => f(v).map(IrArrayLiteralElement::Spread),
                        })
                        .collect_vec();

                    if elements_new.iter().any(|e| e.is_some()) {
                        let new_elements = elements
                            .iter()
                            .zip(elements_new)
                            .map(|(old, new)| new.unwrap_or_else(|| old.clone()))
                            .collect_vec();
                        Some(large.push_expr(IrExpressionLarge::ArrayLiteral(ty.clone(), len.clone(), new_elements)))
                    } else {
                        None
                    }
                }
                IrExpressionLarge::StructLiteral(ty, fields) => build_nary(f, fields)
                    .map(|new| IrExpressionLarge::StructLiteral(ty.clone(), new))
                    .map(|e| large.push_expr(e)),
                &IrExpressionLarge::EnumLiteral(ref ty, variant, ref payload) => {
                    if let Some(payload) = payload {
                        build_unary!(|payload| large.push_expr(IrExpressionLarge::EnumLiteral(
                            ty.clone(),
                            variant,
                            Some(payload)
                        )))
                    } else {
                        None
                    }
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    build_binary!(|base, index| large.push_expr(IrExpressionLarge::ArrayIndex { base, index }))
                }
                IrExpressionLarge::ArraySlice { base, start, len } => {
                    build_binary!(|base, start| large.push_expr(IrExpressionLarge::ArraySlice {
                        base,
                        start,
                        len: len.clone()
                    }))
                }
                &IrExpressionLarge::TupleIndex { ref base, index } => {
                    build_unary!(|base| large.push_expr(IrExpressionLarge::TupleIndex { base, index }))
                }
                &IrExpressionLarge::StructField { ref base, field } => {
                    build_unary!(|base| large.push_expr(IrExpressionLarge::StructField { base, field }))
                }
                IrExpressionLarge::EnumTag { base } => {
                    build_unary!(|base| large.push_expr(IrExpressionLarge::EnumTag { base }))
                }
                &IrExpressionLarge::EnumPayload { ref base, variant } => {
                    build_unary!(|base| large.push_expr(IrExpressionLarge::EnumPayload { base, variant }))
                }
                IrExpressionLarge::ToBits(ty, inner) => {
                    build_unary!(|inner| large.push_expr(IrExpressionLarge::ToBits(ty.clone(), inner)))
                }
                IrExpressionLarge::FromBits(ty, inner) => {
                    build_unary!(|inner| large.push_expr(IrExpressionLarge::FromBits(ty.clone(), inner)))
                }
                IrExpressionLarge::ExpandIntRange(ty, inner) => {
                    build_unary!(|inner| large.push_expr(IrExpressionLarge::ExpandIntRange(ty.clone(), inner)))
                }
                IrExpressionLarge::ConstrainIntRange(ty, inner) => {
                    build_unary!(|inner| large.push_expr(IrExpressionLarge::ConstrainIntRange(ty.clone(), inner)))
                }
            },
        };

        match self_new {
            None => f(self),
            Some(self_new) => match f(&self_new) {
                None => Some(self_new),
                Some(final_new) => Some(final_new),
            },
        }
    }
}

impl Polarized<IrSignal> {
    pub fn as_expression(self, large: &mut IrLargeArena) -> IrExpression {
        let Polarized { inverted, signal } = self;
        let signal = IrExpression::Signal(signal);
        if inverted {
            large.push_expr(IrExpressionLarge::BoolNot(signal))
        } else {
            signal
        }
    }
}

impl IrSignalOrVariable {
    pub fn as_expression(self) -> IrExpression {
        match self {
            IrSignalOrVariable::Signal(s) => IrExpression::Signal(s),
            IrSignalOrVariable::Variable(v) => IrExpression::Variable(v),
        }
    }
}

impl From<IrPort> for IrSignalOrVariable {
    fn from(value: IrPort) -> Self {
        IrSignalOrVariable::Signal(value.into())
    }
}

impl From<IrWire> for IrSignalOrVariable {
    fn from(value: IrWire) -> Self {
        IrSignalOrVariable::Signal(value.into())
    }
}
