// use crate::data::compiled::{GenericItemKind, VariableDomain};
// use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
// use crate::front::common::ValueDomain;
// use crate::front::driver::{CompileState, EvalTrueError};
// use crate::front::types::{IntegerTypeInfo, Type};
// use crate::front::values::{ArrayAccessIndex, BoundedRangeInfo, RangeInfo, Value};
// use crate::syntax::ast::{BinaryOp, PortKind, SyncDomain};
// use crate::syntax::pos::Span;
// use crate::util::option_pair;
// use crate::{try_inner, try_opt_result};
// use annotate_snippets::Level;
// use num_bigint::BigInt;
// use num_traits::One;
// 
// // TODO give this a better name
// #[derive(Debug, Copy, Clone)]
// pub enum DomainUserControlled {
//     Target,
//     Source,
//     Both,
// }
// 
// #[derive(Debug, Copy, Clone)]
// pub struct TypeMismatch;
// 
// impl CompileState<'_, '_> {
//     pub fn domain_of_value(&self, span: Span, value: &Value) -> ValueDomain {
//         let diags = self.diags;
// 
//         match value {
//             &Value::Error(e) => ValueDomain::Error(e),
//             &Value::ModulePort(port) => {
//                 match &self.compiled[port].kind {
//                     PortKind::Clock => ValueDomain::Clock,
//                     PortKind::Normal { domain, ty: _ } => ValueDomain::from_domain_kind(domain.clone()),
//                 }
//             }
//             &Value::GenericParameter(param) => {
//                 let param_info = &self.compiled[param];
//                 match param_info.defining_item_kind {
//                     GenericItemKind::Type | GenericItemKind::Module | GenericItemKind::Struct | GenericItemKind::Enum =>
//                         ValueDomain::CompileTime,
//                     GenericItemKind::Function =>
//                         ValueDomain::FunctionBody(param_info.defining_item),
//                 }
//             }
//             &Value::Never => ValueDomain::CompileTime,
//             &Value::Unit => ValueDomain::CompileTime,
//             &Value::Undefined => ValueDomain::CompileTime,
//             &Value::BoolConstant(_) => ValueDomain::CompileTime,
//             &Value::IntConstant(_) => ValueDomain::CompileTime,
//             &Value::StringConstant(_) => ValueDomain::CompileTime,
//             Value::Range(info) => {
//                 let RangeInfo { start_inc: start_inclusive, end_inc: end_inclusive } = info;
// 
//                 let start_inclusive = start_inclusive.as_ref().map(|v| self.domain_of_value(span, v));
//                 let end_inclusive = end_inclusive.as_ref().map(|v| self.domain_of_value(span, v));
// 
//                 match (start_inclusive, end_inclusive) {
//                     (None, None) =>
//                         ValueDomain::CompileTime,
//                     (Some(single), None) | (None, Some(single)) =>
//                         single,
//                     (Some(start_inclusive), Some(end_inclusive)) =>
//                         self.merge_domains(span, &start_inclusive, &end_inclusive),
//                 }
//             }
//             Value::Binary(_, left, right) =>
//                 self.merge_domains(span, &self.domain_of_value(span, left), &self.domain_of_value(span, right)),
//             Value::UnaryNot(inner) =>
//                 self.domain_of_value(span, inner),
// 
//             Value::ArrayAccess { result_ty: _, base, indices } => {
//                 indices.iter().fold(self.domain_of_value(span, base), |acc, index| {
//                     let index_domain = match index {
//                         &ArrayAccessIndex::Error(e) =>
//                             ValueDomain::Error(e),
//                         ArrayAccessIndex::Single(index) =>
//                             self.domain_of_value(span, index),
//                         ArrayAccessIndex::Range(BoundedRangeInfo { start_inc, end_inc }) =>
//                             self.merge_domains(span, &self.domain_of_value(span, start_inc), &self.domain_of_value(span, end_inc)),
//                     };
//                     self.merge_domains(span, &acc, &index_domain)
//                 })
//             }
//             Value::ArrayLiteral { result_ty: _, operands: ref operands_mixed } => {
//                 operands_mixed.iter().fold(ValueDomain::CompileTime, |acc, value| {
//                     self.merge_domains(span, &acc, &self.domain_of_value(span, &value.value))
//                 })
//             }
// 
//             // TODO just join all argument domains
//             &Value::FunctionReturn(_) =>
//                 ValueDomain::Error(diags.report_todo(span, "domain of function return value")),
//             &Value::Module(_) =>
//                 ValueDomain::Error(diags.report_simple("cannot get domain of module value", span, "module")),
//             &Value::Wire(wire) =>
//                 ValueDomain::from_domain_kind(self.compiled[wire].domain.clone()),
//             &Value::Register(reg) =>
//                 ValueDomain::Sync(self.compiled[reg].domain.clone()),
//             // TODO this is a bit confusing, the origin of the variable matters!
//             &Value::Variable(var) => {
//                 match &self.compiled[var].domain {
//                     VariableDomain::Unknown =>
//                         ValueDomain::Error(diags.report_todo(span, "domain of nontrivial variables")),
//                     VariableDomain::Known(domain) => domain.clone(),
//                 }
//             }
//             &Value::Constant(_) =>
//                 ValueDomain::CompileTime,
//         }
//     }
// 
//     /// Merge two sync domains, as they would be if they were used as part of a single expression.
//     /// Reports an error if this is not possible.
//     pub fn merge_domains(&self, span: Span, left: &ValueDomain, right: &ValueDomain) -> ValueDomain {
//         match (left, right) {
//             // propagate errors
//             (&ValueDomain::Error(e), _) | (_, &ValueDomain::Error(e)) => ValueDomain::Error(e),
//             // const can merge with anything and become that other domain
//             (ValueDomain::CompileTime, other) | (other, ValueDomain::CompileTime) => other.clone(),
//             // sync can merge if both domains match
//             (ValueDomain::Sync(left), ValueDomain::Sync(right)) => {
//                 match sync_domains_equal(left, right) {
//                     Ok(SyncDomainsEqual::Equal) => ValueDomain::Sync(left.clone()),
//                     Ok(SyncDomainsEqual::NotEqual(reason)) => {
//                         let label = format!(
//                             "{}: domains {} and {}",
//                             reason,
//                             self.compiled.sync_kind_to_readable_string(&self.source, &self.parsed, &ValueDomain::Sync(left.clone())),
//                             self.compiled.sync_kind_to_readable_string(&self.source, &self.parsed, &ValueDomain::Sync(right.clone())),
//                         );
//                         let e = self.diags.report_simple("cannot merge different sync domains", span, label);
//                         ValueDomain::Error(e)
//                     }
//                     Err(e) => ValueDomain::Error(e),
//                 }
//             }
//             // function can marge if both domains match
//             (ValueDomain::FunctionBody(left), ValueDomain::FunctionBody(right)) => {
//                 if left == right {
//                     ValueDomain::FunctionBody(left.clone())
//                 } else {
//                     let e = self.diags.report_simple("cannot merge different function body domains", span, "function body domains");
//                     ValueDomain::Error(e)
//                 }
//             }
//             // failed merges
//             (ValueDomain::FunctionBody(_), _) | (_, ValueDomain::FunctionBody(_)) => {
//                 ValueDomain::Error(self.diags.report_simple("cannot merge function domain with anything else", span, "sync domain"))
//             }
//             (ValueDomain::Clock, _) | (_, ValueDomain::Clock) => {
//                 ValueDomain::Error(self.diags.report_simple("cannot merge clock domain with anything else", span, "clock domain"))
//             }
//             (ValueDomain::Async, _) | (_, ValueDomain::Async) => {
//                 ValueDomain::Error(self.diags.report_simple("cannot merge async domain with anything else", span, "async domain"))
//             }
//         }
//     }
// 
//     /// Checks whether the source sync domain can be assigned to the target sync domain.
//     /// This is equivalent to checking whether source is more constrained that target.
//     pub fn check_domain_crossing(
//         &self,
//         target_span: Span,
//         target: &ValueDomain,
//         source_span: Span,
//         source: &ValueDomain,
//         user_controlled: DomainUserControlled,
//         hint: &str,
//     ) -> Result<(), ErrorGuaranteed> {
//         let diags = self.diags;
// 
//         let invalid_reason = match (target, source) {
//             // propagate errors
//             (&ValueDomain::Error(e), _) | (_, &ValueDomain::Error(e)) =>
//                 return Err(e),
//             // TODO think about a fix for delta cycles caused by clock assignments
//             (ValueDomain::Clock, _) | (_, ValueDomain::Clock) => None,
//             // const target must have const source
//             (ValueDomain::CompileTime, ValueDomain::CompileTime) => None,
//             (ValueDomain::CompileTime, ValueDomain::Async) => Some("async to const"),
//             (ValueDomain::CompileTime, ValueDomain::Sync(_)) => Some("sync to const"),
//             (ValueDomain::CompileTime, ValueDomain::FunctionBody(_)) => Some("function body to const"),
//             // const can be the source of everything
//             (ValueDomain::Async, ValueDomain::CompileTime) => None,
//             (ValueDomain::Sync(_), ValueDomain::CompileTime) => None,
//             (ValueDomain::FunctionBody(_), ValueDomain::CompileTime) => None,
//             // async can be the target of everything
//             (ValueDomain::Async, _) => None,
//             // sync cannot be the target of async
//             (ValueDomain::Sync(_), ValueDomain::Async) => Some("async to sync"),
//             // sync pair is allowed if clock and reset match
//             (ValueDomain::Sync(target), ValueDomain::Sync(source)) => {
//                 let SyncDomain { clock: target_clock, reset: target_reset } = target;
//                 let SyncDomain { clock: source_clock, reset: source_reset } = source;
// 
//                 // TODO equality is _probably_ the wrong operation for this
//                 let value_eq = |a: &Value, b: &Value| {
//                     match (a, b) {
//                         // optimistically assume they match
//                         (&Value::Error(_), _) | (_, &Value::Error(_)) => true,
//                         (a, b) => a == b,
//                     }
//                 };
// 
//                 match (value_eq(target_clock, source_clock), value_eq(target_reset, source_reset)) {
//                     (false, false) => Some("different clock and reset"),
//                     (false, true) => Some("different clock"),
//                     (true, false) => Some("different reset"),
//                     (true, true) => None,
//                 }
//             }
//             // function can only be used in matching function body itself
//             (ValueDomain::FunctionBody(target), ValueDomain::FunctionBody(source)) => {
//                 if target == source {
//                     None
//                 } else {
//                     Some("different function body")
//                 }
//             }
//             (ValueDomain::Sync(_), ValueDomain::FunctionBody(_)) => Some("function body to sync"),
//             (ValueDomain::FunctionBody(_), ValueDomain::Sync(_)) => Some("sync to function body"),
//             (ValueDomain::FunctionBody(_), ValueDomain::Async) => Some("async to function body"),
//         };
// 
//         let (target_level, source_level) = match user_controlled {
//             DomainUserControlled::Target => (Level::Error, Level::Info),
//             DomainUserControlled::Source => (Level::Info, Level::Error),
//             DomainUserControlled::Both => (Level::Error, Level::Error),
//         };
// 
//         if let Some(invalid_reason) = invalid_reason {
//             let err = Diagnostic::new(format!("unsafe domain crossing: {}", invalid_reason))
//                 .add(target_level, target_span, format!("target in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, target)))
//                 .add(source_level, source_span, format!("source in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, source)))
//                 .footer(Level::Help, hint)
//                 .finish();
//             Err(diags.report(err))
//         } else {
//             Ok(())
//         }
//     }
// 
//     pub fn type_of_value(&self, span: Span, value: &Value) -> Type {
//         let diags = self.diags;
//         let value = simplify_value(value.clone());
// 
//         let result = match &value {
//             &Value::Error(e) => Type::Error(e),
//             &Value::GenericParameter(param) =>
//                 self.compiled[param].ty.clone(),
//             &Value::ModulePort(port) => {
//                 match &self.compiled[port].kind {
//                     PortKind::Clock => Type::Clock,
//                     PortKind::Normal { domain: _, ty } => ty.clone(),
//                 }
//             }
//             Value::Never => Type::Never,
//             Value::Unit => Type::Unit,
//             Value::Undefined => Type::Unchecked,
//             Value::BoolConstant(_) => Type::Boolean,
//             Value::IntConstant(value) => {
//                 let range = RangeInfo {
//                     start_inc: Some(Box::new(Value::IntConstant(value.clone()))),
//                     end_inc: Some(Box::new(Value::IntConstant(value.clone()))),
//                 };
//                 Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
//             }
//             Value::StringConstant(_) => Type::String,
//             Value::Range(_) => Type::Range,
//             &Value::Binary(op, ref left, ref right) => {
//                 self.type_of_binary(span, op, &left, &right).unwrap_or_else(Type::Error)
//             }
//             Value::UnaryNot(inner) => {
//                 let inner = self.type_of_value(span, inner);
//                 match inner {
//                     Type::Error(e) => Type::Error(e),
//                     // valid types
//                     Type::Boolean | Type::Bits(_) => inner,
//                     // other types are not allowed
//                     _ => {
//                         let inner_ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &inner);
//                         let title = format!("unary not only works for boolean or bits type, got {}", inner_ty_str);
//                         Type::Error(diags.report_simple(title, span, "for this expression"))
//                     }
//                 }
//             }
//             Value::ArrayAccess { result_ty, base: _, indices: _ } => (**result_ty).clone(),
//             Value::ArrayLiteral { ref result_ty, operands: _ } => (**result_ty).clone(),
//             Value::FunctionReturn(func) => func.ret_ty.clone(),
//             Value::Module(_) => Type::Error(diags.report_todo(span, "type of module")),
//             &Value::Wire(wire) => self.compiled[wire].ty.clone(),
//             &Value::Register(reg) => self.compiled[reg].ty.clone(),
//             &Value::Variable(var) => self.compiled[var].ty.clone(),
//             &Value::Constant(constant) => self.compiled[constant].ty.clone(),
//         };
// 
//         if self.log_type_check {
//             eprintln!(
//                 "type_of_value({}) -> {}",
//                 self.compiled.value_to_readable_str(self.source, self.parsed, &value),
//                 self.compiled.type_to_readable_str(self.source, self.parsed, &result)
//             );
//         }
// 
//         result
//     }
// 
//     fn type_of_binary(&self, origin: Span, op: BinaryOp, left: &Value, right: &Value) -> Result<Type, ErrorGuaranteed> {
//         let diags = self.diags;
// 
//         let left_ty = self.type_of_value(origin, left);
//         let right_ty = self.type_of_value(origin, right);
// 
//         fn option_op(op: BinaryOp, a: &Option<Box<Value>>, b: &Option<Box<Value>>) -> Option<Box<Value>> {
//             match (a, b) {
//                 (Some(a), Some(b)) => {
//                     let value = simplify_value(Value::Binary(op, a.clone(), b.clone()));
//                     Some(Box::new(value))
//                 }
//                 _ => None,
//             }
//         }
// 
//         let result = match op {
//             BinaryOp::Add => {
//                 let left_range = self.require_int_range_ty(origin, &left_ty);
//                 let right_range = self.require_int_range_ty(origin, &right_ty);
// 
//                 let left_range = left_range?;
//                 let right_range = right_range?;
// 
//                 let RangeInfo { start_inc: left_start, end_inc: left_end } = left_range;
//                 let RangeInfo { start_inc: right_start, end_inc: right_end } = right_range;
//                 let range = RangeInfo {
//                     start_inc: option_op(BinaryOp::Add, left_start, right_start),
//                     end_inc: option_op(BinaryOp::Add, left_end, right_end),
//                 };
//                 Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
//             }
//             // this is too much duplication, maybe remove the Sub operator and replace it with negation?
//             BinaryOp::Sub => {
//                 let left_range = self.require_int_range_ty(origin, &left_ty);
//                 let right_range = self.require_int_range_ty(origin, &right_ty);
// 
//                 let left_range = left_range?;
//                 let right_range = right_range?;
// 
//                 let RangeInfo { start_inc: left_start, end_inc: left_end } = left_range;
//                 let RangeInfo { start_inc: right_start, end_inc: right_end } = right_range;
// 
//                 let range = RangeInfo {
//                     start_inc: option_op(BinaryOp::Sub, left_start, &right_end),
//                     end_inc: option_op(BinaryOp::Sub, left_end, &right_start),
//                 };
//                 Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
//             }
//             BinaryOp::Pow => {
//                 let base_range = self.require_int_range_ty(origin, &left_ty);
//                 let exp_range = self.require_int_range_ty(origin, &right_ty);
// 
//                 let base_range = base_range?;
//                 let exp_range = exp_range?;
// 
//                 self.type_of_binary_power(origin, base_range, exp_range)
//             }
//             BinaryOp::CmpLt | BinaryOp::CmpLte | BinaryOp::CmpGt | BinaryOp::CmpGte => {
//                 // TODO emit warning if the resulting boolean is always true or always false
//                 let left_result = self.require_int_range_ty(origin, &left_ty);
//                 let right_result = self.require_int_range_ty(origin, &right_ty);
// 
//                 left_result?;
//                 right_result?;
// 
//                 Type::Boolean
//             }
//             BinaryOp::CmpEq | BinaryOp::CmpNeq => {
//                 // TODO emit warning if the resulting boolean is always true or always false
//                 match (&left_ty, &right_ty) {
//                     (Type::Boolean, Type::Boolean) => {}
//                     (Type::Integer(_), Type::Integer(_)) => {}
//                     (Type::Bits(n_left), Type::Bits(n_right)) => {
//                         // sizes must match
//                         match (n_left, n_right) {
//                             (None, None) => {}
//                             (Some(n_left), Some(n_right)) => {
//                                 let _ = self.require_value_true_for_type_check(None, origin, &Value::Binary(BinaryOp::CmpEq, n_left.clone(), n_right.clone()));
//                             }
//                             (None, Some(_)) | (Some(_), None) => {
//                                 let title = "type mismatch: cannot compare bits when one has a finite size and the other does not";
//                                 return Err(self.diags.report_simple(title, origin, "for this expression"));
//                             }
//                         }
//                     }
//                     _ => {
//                         let left_str = self.compiled.type_to_readable_str(self.source, self.parsed, &left_ty);
//                         let right_str = self.compiled.type_to_readable_str(self.source, self.parsed, &right_ty);
//                         let title = format!("type mismatch: cannot compare types {} and {}", left_str, right_str);
//                         return Err(self.diags.report_simple(title, origin, "for this expression"));
//                     }
//                 }
// 
//                 Type::Boolean
//             }
//             BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
//                 // check that operands are "bits" with the same size
//                 match (&left_ty, &right_ty) {
//                     (Type::Bits(left_size), Type::Bits(right_size)) => {
//                         match (left_size, right_size) {
//                             (None, None) => Type::Bits(None),
//                             (Some(left_size), Some(right_size)) => {
//                                 match self.require_value_true_for_type_check(None, origin, &Value::Binary(BinaryOp::CmpEq, left_size.clone(), right_size.clone())) {
//                                     Ok(()) => Type::Bits(Some(left_size.clone())),
//                                     Err(e) => Type::Error(e),
//                                 }
//                             }
//                             _ => {
//                                 let title = "type mismatch: cannot combine bits when one has a finite size and the other does not";
//                                 Type::Error(self.diags.report_simple(title, origin, "for this expression"))
//                             }
//                         }
//                     }
//                     _ => {
//                         let title = "type mismatch: operands must be bits type for bit operator";
//                         Type::Error(self.diags.report_simple(title, origin, "in this expression"))
//                     }
//                 }
//             }
//             BinaryOp::BoolAnd | BinaryOp::BoolOr | BinaryOp::BoolXor => {
//                 let check_bool_ty = |ty: Type| -> Result<(), ErrorGuaranteed>{
//                     match ty {
//                         Type::Boolean => Ok(()),
//                         _ => {
//                             let title = "type mismatch: expected boolean type for boolean operator";
//                             Err(self.diags.report_simple(title, origin, "for this expression"))
//                         }
//                     }
//                 };
// 
//                 let left_result = check_bool_ty(left_ty);
//                 let right_result = check_bool_ty(right_ty);
// 
//                 left_result?;
//                 right_result?;
//                 Type::Boolean
//             }
//             BinaryOp::Mul => {
//                 let left_range = self.require_int_range_ty(origin, &left_ty);
//                 let right_range = self.require_int_range_ty(origin, &right_ty);
// 
//                 let left_range = left_range?;
//                 let right_range = right_range?;
// 
//                 let RangeInfo { start_inc: left_start, end_inc: left_end } = left_range;
//                 let RangeInfo { start_inc: right_start, end_inc: right_end } = right_range;
// 
//                 let is_non_negative = |value: &Option<Box<Value>>| -> Result<bool, ErrorGuaranteed> {
//                     match value {
//                         Some(value) => {
//                             let cond = Value::Binary(BinaryOp::CmpLte, Box::new(Value::IntConstant(BigInt::ZERO)), value.clone());
//                             Ok(self.try_eval_bool(origin, &cond)?.unwrap_or(false))
//                         }
//                         None => Ok(false),
//                     }
//                 };
// 
//                 let range = if is_non_negative(left_start)? && is_non_negative(right_start)? {
//                     // if both are positive, the range is simple
//                     RangeInfo {
//                         start_inc: option_op(BinaryOp::Mul, left_start, right_start),
//                         end_inc: option_op(BinaryOp::Mul, left_end, right_end),
//                     }
//                 } else {
//                     // TODO this can be improved a lot:
//                     // * if either of them does not cross zero the result range is knowable
//                     // * worst case, just fallback to emitting (min(..., ...)..(max(..., ...))
//                     RangeInfo::UNBOUNDED
//                 };
// 
//                 Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
//             }
//             BinaryOp::Mod => {
//                 let left_range = self.require_int_range_ty(origin, &left_ty);
//                 let right_range = self.require_int_range_ty(origin, &right_ty);
// 
//                 let left_range = left_range?;
//                 let right_range = right_range?;
// 
//                 let RangeInfo { start_inc: _, end_inc: _ } = left_range;
//                 let RangeInfo { start_inc: right_start, end_inc: _ } = right_range;
// 
//                 // TODO allow non-positive module operations
//                 let right_start = right_start
//                     .as_ref()
//                     .ok_or_else(|| diags.report_todo(origin, "modulo with non-positive right hand side"))?;
//                 let cond = Value::Binary(BinaryOp::CmpLt, Box::new(Value::IntConstant(BigInt::ZERO)), right_start.clone());
//                 match self.try_eval_bool_true(origin, &cond) {
//                     Ok(()) => {}
//                     Err(e) => {
//                         let _: EvalTrueError = e;
//                         diags.report_todo(origin, "modulo with non-positive right hand side");
//                     }
//                 }
// 
//                 // TODO we can do better than this, eg. if left is smaller than right we can keep the left range
//                 let range = RangeInfo {
//                     start_inc: Some(Box::new(Value::IntConstant(BigInt::ZERO))),
//                     end_inc: Some(Box::new(Value::Binary(BinaryOp::Sub, Box::new(right.clone()), Box::new(Value::IntConstant(BigInt::one()))))),
//                 };
//                 Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
//             }
//             BinaryOp::In => {
//                 // TODO emit warning if the result is always true or false
//                 let left_result = self.require_int_ty(origin, &left_ty);
//                 let right_result = match right_ty {
//                     Type::Range => Ok(()),
//                     _ => {
//                         let title = "type mismatch: expected range type for right operand of 'in' operator";
//                         Err(self.diags.report_simple(title, origin, "for this expression"))
//                     }
//                 };
// 
//                 left_result?;
//                 right_result?;
// 
//                 Type::Boolean
//             }
// 
//             // TODO careful with signs and zero
//             BinaryOp::Div =>
//                 Type::Error(self.diags.report_todo(origin, "type of division")),
//             // TODO we even want to keep these, or just use division/multiplication and powers?
//             //   maybe these apply to bits, not integers?
//             BinaryOp::Shl | BinaryOp::Shr =>
//                 Type::Error(self.diags.report_todo(origin, "type of shift operators")),
//         };
// 
//         Ok(result)
//     }
// 
//     fn type_of_binary_power(&self, origin: Span, base_range: &RangeInfo<Box<Value>>, exp_range: &RangeInfo<Box<Value>>) -> Type {
//         let diags = self.diags;
// 
//         let RangeInfo { start_inc: base_start, end_inc: _ } = base_range;
//         let RangeInfo { start_inc: exp_start, end_inc: _ } = exp_range;
// 
//         // check that exponent is non-negative
//         let exp_start = match exp_start {
//             None => {
//                 let exp_str = self.compiled.range_to_readable_str(self.source, self.parsed, &exp_range);
//                 let title = format!("power exponent cannot be negative, got range without lower bound {:?}", exp_str);
//                 let e = diags.report_simple(title, origin, "while checking this expression");
//                 return Type::Error(e);
//             }
//             Some(exp_start) => exp_start
//         };
//         let cond = Value::Binary(BinaryOp::CmpLte, Box::new(Value::IntConstant(BigInt::ZERO)), exp_start.clone());
//         match self.try_eval_bool_true(origin, &cond) {
//             Ok(()) => {}
//             Err(e) => {
//                 let cond_str = self.compiled.value_to_readable_str(self.source, self.parsed, &cond);
//                 let title = format!("power exponent cannot be negative, check failed: value {} {}", cond_str, e.to_message());
//                 let e = diags.report_simple(title, origin, "while checking this expression");
//                 return Type::Error(e);
//             }
//         }
// 
//         // compute facts about the ranges
//         let eval_known_optional = |op: BinaryOp, left: Option<Box<Value>>, right: Option<Box<Value>>| {
//             let result = match (left, right) {
//                 (Some(left), Some(right)) => {
//                     let base_positive = Value::Binary(op, left, right);
//                     self.try_eval_bool(origin, &base_positive)
//                 }
//                 (None, _) | (_, None) => Ok(None),
//             };
//             match result {
//                 Ok(Some(b)) => Ok(b),
//                 Ok(None) => Ok(false),
//                 Err(e) => Err(e),
//             }
//         };
//         macro_rules! throw_type {
//             ($e:expr) => {
//                 match $e {
//                     Ok(v) => v,
//                     Err(e) => return Type::Error(e),
//                 }
//             };
//         }
// 
//         let known_base_positive = throw_type!(
//             eval_known_optional(BinaryOp::CmpLt, Some(Box::new(Value::IntConstant(BigInt::ZERO))), base_start.clone()));
//         let known_base_negative = throw_type!(
//             eval_known_optional(BinaryOp::CmpGt, Some(Box::new(Value::IntConstant(BigInt::ZERO))), base_start.clone()));
//         let known_base_non_zero = known_base_positive | known_base_negative;
//         let known_exponent_positive = throw_type!(
//             eval_known_optional(BinaryOp::CmpLt, Some(Box::new(Value::IntConstant(BigInt::ZERO))), Some(exp_start.clone())));
// 
//         // check that `0 ** 0` can't happen
//         // TODO at some point this should be lifted into the statement prover,
//         //   users should be allowed to check for this themselves!
//         if !(known_base_non_zero || known_exponent_positive) {
//             let e = diags.report_simple(
//                 "power expression where base and exponent can both be zero is not allowed",
//                 origin,
//                 "while checking this expression",
//             );
//             return Type::Error(e);
//         }
// 
//         // some simple range result logic
//         // TODO this can be improved:
//         // * is exp is 1, the result range is the base range
//         // * if exp is 0, the result range is 1
//         // * if exp is even, result is non-negative
//         // * if exp is odd, result is the same sign as base
//         let range = if known_base_positive {
//             // (>0) ** (>=0) = (>0)
//             RangeInfo { start_inc: Some(BigInt::one()), end_inc: None }
//         } else {
//             RangeInfo { start_inc: None, end_inc: None }
//         };
// 
//         let range = range.map_inner(|x| Box::new(Value::IntConstant(x)));
//         Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
//     }
// 
//     fn require_int_ty<'a>(&self, span: Span, ty: &'a Type) -> Result<&'a Value, ErrorGuaranteed> {
//         match ty {
//             &Type::Error(e) => Err(e),
//             Type::Integer(IntegerTypeInfo { range }) => Ok(range),
//             _ => Err(self.diags.report_simple("expected integer type", span, "for this expression")),
//         }
//     }
// 
//     fn require_int_range_ty<'a>(&self, span: Span, ty: &'a Type) -> Result<&'a RangeInfo<Box<Value>>, ErrorGuaranteed> {
//         match self.require_int_ty(span, ty) {
//             Ok(range) => self.require_int_range_direct(span, range),
//             Err(e) => Err(e),
//         }
//     }
// 
//     pub fn require_int_range_direct<'a>(&self, span: Span, range: &'a Value) -> Result<&'a RangeInfo<Box<Value>>, ErrorGuaranteed> {
//         match range {
//             &Value::Error(e) => Err(e),
//             Value::Range(range) => Ok(range),
//             _ => Err(self.diags.report_todo(span, "indirect integer range")),
//         }
//     }
// 
//     // TODO double-check that all of these type-checking functions do their error handling correctly
//     //   and don't short-circuit unnecessarily, preventing multiple errors
//     // TODO move error formatting out of this function, it depends too much on the context and spans are not always available
//     pub fn require_type_contains_value(&self, span_ty: Option<Span>, span_value: Span, ty: &Type, value: &Value) -> Result<(), ErrorGuaranteed> {
//         let ty_value = self.type_of_value(span_value, value);
// 
//         match self.require_type_match(span_ty, ty, span_value, &ty_value, true) {
//             Ok(Ok(())) => Ok(()),
//             Ok(Err(e)) => Err(e),
//             Err(e) => {
//                 let _: TypeMismatch = e;
//                 let ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, ty);
//                 let value_str = self.compiled.value_to_readable_str(self.source, self.parsed, value);
//                 let value_ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &ty_value);
// 
//                 let title = format!("type mismatch: value `{}` with type `{}` does not fit in type `{}`", value_str, value_ty_str, ty_str);
//                 let err = Diagnostic::new(title)
//                     .add_error(span_value, "value used here")
//                     .add_info_maybe(span_ty, "type defined here")
//                     .finish();
//                 Err(self.diags.report(err))
//             }
//         }
//     }
// 
//     pub fn require_type_match(&self, target_span: Option<Span>, target_ty: &Type, source_span: Span, source_ty: &Type, allow_subtype: bool) -> Result<Result<(), ErrorGuaranteed>, TypeMismatch> {
//         match (target_ty, source_ty) {
//             // propagate errors, we can't just silently ignore them:
//             //   downstream compiler code might actually depend on the type matching
//             (&Type::Error(e), _) | (_, &Type::Error(e)) => Ok(Err(e)),
// 
//             // equal types are always fine
//             _ if target_ty == source_ty => Ok(Ok(())),
// 
//             // any contains everything
//             (&Type::Any, _other) => {
//                 if allow_subtype {
//                     Ok(Ok(()))
//                 } else {
//                     Err(TypeMismatch)
//                 }
//             }
// 
//             // unchecked and never cast into anything
//             (_, Type::Unchecked) => Ok(Ok(())),
//             (_, Type::Never) => Ok(Ok(())),
// 
//             // basic types that only contain themselves
//             (Type::Unit, Type::Unit) => Ok(Ok(())),
//             // TODO differentiate between arbitrary open, half-open, ...
//             (Type::Range, Type::Range) => Ok(Ok(())),
//             (Type::Boolean, Type::Boolean) => Ok(Ok(())),
// 
//             // array type matches if inner type is the same (not just a subtype) and the lengths are the same
//             (Type::Array(inner_left, len_left), Type::Array(inner_right, len_right)) => {
//                 // array types are invariant, so the inner types must match exactly
//                 try_inner!(self.require_type_match(None, inner_left, source_span, inner_right, false)?);
// 
//                 // lengths must match
//                 let cond = Value::Binary(BinaryOp::CmpEq, len_left.clone(), len_right.clone());
//                 try_inner!(self.require_value_true_for_type_check(target_span, source_span, &cond));
// 
//                 Ok(Ok(()))
//             }
// 
//             // integer range check
//             (
//                 Type::Integer(IntegerTypeInfo { range: target_range }),
//                 Type::Integer(IntegerTypeInfo { range: source_range })
//             ) => {
//                 // check that the value fits in the required range
//                 let RangeInfo { start_inc: target_start_inc, end_inc: target_end_inc } =
//                     try_inner!(self.require_int_range_direct(target_span.unwrap_or(source_span), target_range));
//                 let RangeInfo { start_inc: source_start_inc, end_inc: source_end_inc } =
//                     try_inner!(self.require_int_range_direct(source_span, source_range));
// 
//                 if allow_subtype {
//                     match (target_start_inc, source_start_inc) {
//                         (None, None | Some(_)) => {}
//                         (Some(target_start_inc), Some(source_start_inc)) => {
//                             let cond = Value::Binary(BinaryOp::CmpLte, target_start_inc.clone(), source_start_inc.clone());
//                             try_inner!(self.require_value_true_for_type_check(target_span, source_span, &cond));
//                         }
//                         (Some(_), None) => return Err(TypeMismatch),
//                     }
// 
//                     match (target_end_inc, source_end_inc) {
//                         (None, None | Some(_)) => {}
//                         (Some(target_end_inc), Some(source_end_inc)) => {
//                             let cond = Value::Binary(BinaryOp::CmpGte, target_end_inc.clone(), source_end_inc.clone());
//                             try_inner!(self.require_value_true_for_type_check(target_span, source_span, &cond));
//                         }
//                         (Some(_), None) => return Err(TypeMismatch),
//                     }
//                 } else {
//                     return Err(TypeMismatch);
//                 }
// 
//                 Ok(Ok(()))
//             }
// 
//             _ => Err(TypeMismatch)
//         }
//     }
// 
//     pub fn require_value_true_for_range(&self, span_range: Span, value: &Value) -> Result<(), ErrorGuaranteed> {
//         self.try_eval_bool_true(span_range, value).map_err(|e| {
//             let value_str = self.compiled.value_to_readable_str(self.source, self.parsed, value);
//             let title = format!("range valid check failed: value {} {}", value_str, e.to_message());
//             let err = Diagnostic::new(title)
//                 .add_error(span_range, "when checking that this range is non-decreasing")
//                 .finish();
//             self.diags.report(err).into()
//         })
//     }
// 
//     pub fn require_value_true_for_type_check(&self, span_ty: Option<Span>, span_value: Span, value: &Value) -> Result<(), ErrorGuaranteed> {
//         self.try_eval_bool_true(span_value, value).map_err(|e| {
//             let value_str = self.compiled.value_to_readable_str(self.source, self.parsed, value);
//             let title = format!("type check failed: value {} {}", value_str, e.to_message());
//             // TODO include the type of the value and the target type, with generics it's not always obvious
//             let err = Diagnostic::new(title)
//                 .add_error(span_value, "when type checking this value")
//                 .add_info_maybe(span_ty, "against this type")
//                 .finish();
//             self.diags.report(err).into()
//         })
//     }
// 
//     pub fn try_eval_bool_true(&self, origin: Span, value: &Value) -> Result<(), EvalTrueError> {
//         match self.try_eval_bool(origin, value) {
//             Err(e) => {
//                 // accept errors as correct to prevent further downstream errors
//                 let _: ErrorGuaranteed = e;
//                 Ok(())
//             }
//             Ok(Some(true)) => Ok(()),
//             Ok(Some(false)) => Err(EvalTrueError::False),
//             Ok(None) => Err(EvalTrueError::Unknown),
//         }
//     }
// 
//     // TODO what is the general algorithm for this? equivalence graphs?
//     // TODO add boolean-proving cache
//     // TODO it it possible to keep boolean proving and type inference separate? it probably is
//     //   if we don't allow user-defined type selections
//     //   even then, we can probably brute-force our way through those relatively easily
//     // TODO convert lte/gte into +1/-1 fixes instead?
//     // TODO convert inclusive/exclusive into +1/-1 fixes instead?
//     // TODO check lt, lte, gt, gte, ... all together elegantly
//     // TODO return true for vacuous truths, eg. comparisons between empty ranges?
//     pub fn try_eval_bool(&self, origin: Span, value: &Value) -> Result<Option<bool>, ErrorGuaranteed> {
//         if self.log_type_check {
//             eprintln!(
//                 "try_eval_bool({})",
//                 self.compiled.value_to_readable_str(self.source, self.parsed, value),
//             );
//         }
//         let result = self.try_eval_bool_inner(origin, value);
//         if self.log_type_check {
//             eprintln!(
//                 "try_eval_bool({}) -> {:?}",
//                 self.compiled.value_to_readable_str(self.source, self.parsed, value),
//                 result
//             );
//         }
//         result
//     }
// 
//     // TODO if range ends are themselves params with ranges, assume the worst case
//     //   although that misses things like (n < n+1)
//     pub fn try_eval_bool_inner(&self, origin: Span, value: &Value) -> Result<Option<bool>, ErrorGuaranteed> {
//         if self.log_type_check {
//             eprintln!(
//                 "try_eval_bool_inner({})",
//                 self.compiled.value_to_readable_str(self.source, self.parsed, value),
//             );
//         }
// 
//         // TODO this is wrong, we should be returning None a lot more, eg. if the ranges of the operands are not tight
//         match *value {
//             Value::Error(e) => return Err(e),
//             Value::Binary(binary_op, ref left_raw, ref right_raw) => {
//                 let (comparison, left, right) = match binary_op {
//                     BinaryOp::CmpEq => {
//                         // special case: symbolic equality check
//                         // TODO handle error correctly here
//                         if simplify_value((**left_raw).clone()) == simplify_value((**right_raw).clone()) {
//                             return Ok(Some(true));
//                         }
//                         (ReducedComparison::Eq, left_raw, right_raw)
//                     }
//                     BinaryOp::CmpNeq => (ReducedComparison::Neq, left_raw, right_raw),
//                     BinaryOp::CmpLt => (ReducedComparison::Lt, left_raw, right_raw),
//                     BinaryOp::CmpLte => (ReducedComparison::Lte, left_raw, right_raw),
//                     BinaryOp::CmpGt => (ReducedComparison::Lt, right_raw, left_raw),
//                     BinaryOp::CmpGte => (ReducedComparison::Lte, right_raw, left_raw),
//                     _ => return Ok(None),
//                 };
// 
//                 let left_ty = self.type_of_value(origin, left);
//                 let right_ty = self.type_of_value(origin, right);
//                 if self.log_type_check {
//                     eprintln!("comparing values with op {:?}", binary_op);
//                     eprintln!("  {:?}", left_ty);
//                     eprintln!("  {:?}", right_ty);
//                 }
//                 let left_range = self.require_int_range_ty(origin, &left_ty)?;
//                 let right_range = self.require_int_range_ty(origin, &right_ty)?;
//                 if self.log_type_check {
//                     eprintln!("comparing ranges with op {:?}", binary_op);
//                     eprintln!("  {:?}", left_range);
//                     eprintln!("  {:?}", right_range);
//                 }
// 
//                 return match comparison {
//                     // TODO eq and neq (based on ranges)
//                     ReducedComparison::Eq => Ok(None),
//                     ReducedComparison::Neq => Ok(None),
//                     ReducedComparison::Lt => {
//                         let left_end = match left_range.end_inc.as_ref() {
//                             None => return Ok(Some(false)),
//                             Some(left_end) => try_opt_result!(value_as_int(left_end)),
//                         };
//                         let right_start = match right_range.start_inc.as_ref() {
//                             None => return Ok(Some(false)),
//                             Some(right_start) => try_opt_result!(value_as_int(right_start)),
//                         };
//                         Ok(Some(left_end < right_start))
//                     }
//                     ReducedComparison::Lte => {
//                         let left_end = match left_range.end_inc.as_ref() {
//                             None => return Ok(Some(false)),
//                             Some(left_end) => try_opt_result!(value_as_int(left_end)),
//                         };
//                         let right_start = match right_range.start_inc.as_ref() {
//                             None => return Ok(Some(false)),
//                             Some(right_start) => try_opt_result!(value_as_int(right_start)),
//                         };
//                         Ok(Some(left_end <= right_start))
//                     }
//                 };
//             }
//             // TODO support more values
//             _ => {}
//         }
//         Ok(None)
//     }
// }
// 
// pub fn simplify_value(value: Value) -> Value {
//     match value {
//         Value::Binary(op, left, right) => {
//             let left = simplify_value(*left);
//             let right = simplify_value(*right);
// 
//             if let Some((left, right)) = option_pair(value_as_int(&left), value_as_int(&right)) {
//                 match op {
//                     BinaryOp::Add => return Value::IntConstant(left + right),
//                     BinaryOp::Sub => return Value::IntConstant(left - right),
//                     BinaryOp::Mul => return Value::IntConstant(left * right),
//                     _ => {}
//                 }
//             }
// 
//             Value::Binary(op, Box::new(left), Box::new(right))
//         }
//         Value::UnaryNot(inner) => {
//             let inner = simplify_value(*inner);
//             Value::UnaryNot(Box::new(inner))
//         }
//         // TODO at least recursively call simplify
//         value => value,
//     }
// }
// 
// // TODO return error if value is error?
// pub fn value_as_int(value: &Value) -> Option<BigInt> {
//     match simplify_value(value.clone()) {
//         Value::IntConstant(value) => Some(value),
//         _ => None,
//     }
// }
// 
// pub enum SyncDomainsEqual {
//     Equal,
//     NotEqual(&'static str),
// }
// 
// pub fn sync_domains_equal(left: &SyncDomain<Value>, right: &SyncDomain<Value>) -> Result<SyncDomainsEqual, ErrorGuaranteed> {
//     let SyncDomain { clock: target_clock, reset: target_reset } = left;
//     let SyncDomain { clock: source_clock, reset: source_reset } = right;
// 
//     let value_eq = |a: &Value, b: &Value| {
//         match (a, b) {
//             // optimistically assume they match
//             (&Value::Error(e), _) | (_, &Value::Error(e)) => Err(e),
//             // TODO equality is _probably_ the wrong operation for this
//             (a, b) => Ok(a == b),
//         }
//     };
// 
//     // we're intentionally not emitting any error if _either_ of them is an error already, to prevent confusion
//     match (value_eq(target_clock, source_clock)?, value_eq(target_reset, source_reset)?) {
//         (false, false) => Ok(SyncDomainsEqual::NotEqual("different clock and reset")),
//         (false, true) => Ok(SyncDomainsEqual::NotEqual("different clock")),
//         (true, false) => Ok(SyncDomainsEqual::NotEqual("different reset")),
//         (true, true) => Ok(SyncDomainsEqual::Equal),
//     }
// }
// 
// #[derive(Debug, Copy, Clone)]
// enum ReducedComparison {
//     Eq,
//     Neq,
//     Lt,
//     Lte,
// }