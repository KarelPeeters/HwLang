// use crate::data::compiled::{CompiledDatabase, CompiledStage, Constant, GenericValueParameter, Item, ModulePort, Register, Variable, Wire};
// use crate::data::diagnostic::ErrorGuaranteed;
// use crate::front::common::{GenericContainer, GenericMap};
// use crate::front::types::{NominalTypeUnique, Type};
// use crate::syntax::ast::{ArrayLiteralElement, BinaryOp};
// use itertools::Itertools;
// use num_bigint::BigInt;
// 
// // TODO should all values have types? or can eg. ints just be free abstract objects?
// // TODO during compilation, have a "value" wrapper that lazily computes the content and type to break up cycles
// // TODO should all values (and types) have (optional) origin spans for easier error messages?
// // TODO Eq impl is a bit suspicious, remove it and replace it by named functions, eg. is_same_value
// // TODO attach a span to each value? that kills both interning and compiler-value-constructing though
// #[derive(Debug, Clone, Eq, PartialEq, Hash)]
// pub enum Value {
//     // error
//     Error(ErrorGuaranteed),
// 
//     // parameters
//     GenericParameter(GenericValueParameter),
//     ModulePort(ModulePort),
// 
//     // basic
//     Never,
//     Unit,
//     Undefined,
//     BoolConstant(bool),
//     IntConstant(BigInt),
//     StringConstant(String),
//     // TODO long-term this should become a standard struct instead of compiler magic
//     Range(RangeInfo<Box<Value>>),
//     // TODO this BinaryOp should probably be separate from the ast one
//     Binary(BinaryOp, Box<Value>, Box<Value>),
//     UnaryNot(Box<Value>),
//     ArrayAccess {
//         result_ty: Box<Type>,
//         base: Box<Value>,
//         indices: Vec<ArrayAccessIndex<Box<Value>>>,
//     },
//     ArrayLiteral {
//         result_ty: Box<Type>,
//         operands: Vec<ArrayLiteralElement<Value>>,
//     },
// 
//     // structures
//     // TODO functions are represented very strangely, double-check if this makes sense
//     FunctionReturn(FunctionReturnValue),
//     Module(ModuleValueInfo),
//     // Struct(StructValue),
//     // Tuple(TupleValue),
//     // Enum(EnumValue),
// 
//     // variables
//     // TODO how should these behave under generic substitution?
//     Wire(Wire),
//     Register(Register),
//     Variable(Variable),
//     Constant(Constant),
// }
// 
// #[derive(Debug, Clone, Eq, PartialEq, Hash)]
// pub enum ArrayAccessIndex<V> {
//     Error(ErrorGuaranteed),
//     Single(V),
//     Range(BoundedRangeInfo<V>),
// }
// 
// /// Both start and end are inclusive.
// /// This is convenient for arithmetic range calculations.
// #[derive(Debug, Clone, Eq, PartialEq, Hash)]
// pub struct RangeInfo<V> {
//     pub start_inc: Option<V>,
//     pub end_inc: Option<V>,
// }
// 
// #[derive(Debug, Clone, Eq, PartialEq, Hash)]
// pub struct BoundedRangeInfo<V> {
//     pub start_inc: V,
//     pub end_inc: V,
// }
// 
// // TODO double check which fields should be used for eq and hash
// #[derive(Debug, Clone, Eq, PartialEq, Hash)]
// pub struct FunctionReturnValue {
//     pub item: Item,
//     pub ret_ty: Type,
// }
// 
// #[derive(Debug, Clone, Eq, PartialEq, Hash)]
// pub struct ModuleValueInfo {
//     // TODO should this be here or not?
//     pub nominal_type_unique: NominalTypeUnique,
//     pub ports: Vec<ModulePort>,
// }
// 
// impl<V> RangeInfo<V> {
//     pub const UNBOUNDED: RangeInfo<V> = RangeInfo {
//         start_inc: None,
//         end_inc: None,
//     };
// 
//     pub fn map_inner<U>(self, mut f: impl FnMut(V) -> U) -> RangeInfo<U> {
//         RangeInfo {
//             start_inc: self.start_inc.map(&mut f),
//             end_inc: self.end_inc.map(&mut f),
//         }
//     }
// }
// 
// impl GenericContainer for Value {
//     type Result = Value;
// 
//     fn replace_generics<S: CompiledStage>(
//         &self,
//         compiled: &mut CompiledDatabase<S>,
//         map: &GenericMap,
//     ) -> Value {
//         match *self {
//             Value::Error(e) => Value::Error(e),
// 
//             Value::GenericParameter(param) =>
//                 param.replace_generics(compiled, map),
//             Value::ModulePort(module_port) => {
//                 match map.module_port.get(&module_port) {
//                     Some(value) => value.clone(),
//                     None => Value::ModulePort(module_port),
//                 }
//             }
// 
//             Value::Unit => Value::Unit,
//             Value::Never => Value::Never,
//             Value::Undefined => Value::Undefined,
// 
//             Value::BoolConstant(b) => Value::BoolConstant(b),
//             Value::IntConstant(ref info) => Value::IntConstant(info.clone()),
//             Value::StringConstant(ref info) => Value::StringConstant(info.clone()),
//             Value::Range(ref info) => Value::Range(info.replace_generics(compiled, map)),
//             Value::Binary(op, ref left, ref right) => {
//                 Value::Binary(
//                     op,
//                     Box::new(left.replace_generics(compiled, map)),
//                     Box::new(right.replace_generics(compiled, map)),
//                 )
//             }
//             Value::UnaryNot(ref inner) =>
//                 Value::UnaryNot(Box::new(inner.replace_generics(compiled, map))),
//             Value::ArrayAccess { ref result_ty, ref base, ref indices } => {
//                 Value::ArrayAccess {
//                     result_ty: Box::new(result_ty.replace_generics(compiled, map)),
//                     base: Box::new(base.replace_generics(compiled, map)),
//                     indices: indices.iter().map(|v| v.replace_generics(compiled, map)).collect_vec(),
//                 }
//             }
//             Value::ArrayLiteral { ref result_ty, operands: ref operands_mixed } => {
//                 Value::ArrayLiteral {
//                     result_ty: Box::new(result_ty.replace_generics(compiled, map)),
//                     operands: operands_mixed.iter().map(|v| {
//                         ArrayLiteralElement {
//                             spread: v.spread,
//                             value: v.value.replace_generics(compiled, map),
//                         }
//                     }).collect_vec(),
//                 }
//             }
// 
//             Value::FunctionReturn(ref func) => {
//                 Value::FunctionReturn(FunctionReturnValue {
//                     item: func.item,
//                     ret_ty: func.ret_ty.replace_generics(compiled, map),
//                 })
//             }
//             Value::Module(ref info) => Value::Module(info.replace_generics(compiled, map)),
// 
//             // TODO replace generics, similar to ports!
//             //   ideally we switch to something where we don't have to replace generics,
//             //   and can just use the original value + a substitution context
//             Value::Wire(w) => Value::Wire(w),
//             Value::Register(r) => Value::Register(r),
//             Value::Variable(v) => Value::Variable(v),
//             Value::Constant(c) => Value::Constant(c),
//         }
//     }
// }
// 
// impl GenericContainer for ArrayAccessIndex<Box<Value>> {
//     type Result = ArrayAccessIndex<Box<Value>>;
// 
//     fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
//         match self {
//             ArrayAccessIndex::Error(e) => ArrayAccessIndex::Error(e.clone()),
//             ArrayAccessIndex::Single(v) => ArrayAccessIndex::Single(Box::new(v.replace_generics(compiled, map))),
//             ArrayAccessIndex::Range(r) => ArrayAccessIndex::Range(r.replace_generics(compiled, map)),
//         }
//     }
// }
// 
// impl GenericContainer for RangeInfo<Box<Value>> {
//     type Result = RangeInfo<Box<Value>>;
// 
//     fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
//         RangeInfo {
//             start_inc: self.start_inc.as_ref().map(|v| Box::new(v.replace_generics(compiled, map))),
//             end_inc: self.end_inc.as_ref().map(|v| Box::new(v.replace_generics(compiled, map))),
//         }
//     }
// }
// 
// impl GenericContainer for BoundedRangeInfo<Box<Value>> {
//     type Result = BoundedRangeInfo<Box<Value>>;
// 
//     fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
//         BoundedRangeInfo {
//             start_inc: Box::new(self.start_inc.replace_generics(compiled, map)),
//             end_inc: Box::new(self.end_inc.replace_generics(compiled, map)),
//         }
//     }
// }
// 
// impl GenericContainer for ModuleValueInfo {
//     type Result = ModuleValueInfo;
// 
//     fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
//         ModuleValueInfo {
//             nominal_type_unique: self.nominal_type_unique.clone(),
//             ports: self.ports.iter().map(|p| p.replace_generics(compiled, map)).collect_vec(),
//         }
//     }
// }