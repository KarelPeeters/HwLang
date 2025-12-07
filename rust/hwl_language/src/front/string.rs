use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::DiagResult;
use crate::front::flow::{Flow, FlowHardware};
use crate::front::item::ElaborationArenas;
use crate::front::range::{ClosedIncRange, IncRange};
use crate::front::scope::Scope;
use crate::front::types::{HardwareType, Type};
use crate::front::value::{
    CompileCompoundValue, CompileValue, EnumValue, HardwareValue, MixedCompoundValue, RangeEnd, RangeValue,
    SimpleCompileValue, StructValue, Value,
};
use crate::mid::ir::{
    IrBlock, IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrIntCompareOp, IrIntegerRadix,
    IrLargeArena, IrStatement, IrStringPiece, IrStringSubstitution, IrType, IrVariable, IrVariableInfo,
};
use crate::syntax::ast::{Expression, StringPiece};
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::token::{TOKEN_STR_BUILTIN, apply_string_literal_escapes};
use crate::util::big_int::{BigInt, BigUint};
use crate::util::iter::IterExt;
use hwl_util::swrite;
use itertools::{Itertools, enumerate, zip_eq};
use std::borrow::Cow;
use std::sync::Arc;

struct StringBuilder {
    pieces: Vec<StringPiece<String, HardwareValue>>,
}

impl StringBuilder {
    pub fn new() -> Self {
        Self { pieces: vec![] }
    }

    pub fn push_str<'s>(&mut self, value: impl Into<Cow<'s, str>>) {
        let value = value.into();
        if let Some(StringPiece::Literal(last)) = self.pieces.last_mut() {
            // concatenate adjacent literals to canonicalize
            last.push_str(&value);
        } else {
            // just push a new piece
            self.pieces.push(StringPiece::Literal(value.into_owned()))
        }
    }

    pub fn push_value(&mut self, elab: &ElaborationArenas, value: &Value) {
        match value {
            Value::Simple(value) => self.push_str(value.value_string(elab)),
            Value::Compound(value) => match value {
                MixedCompoundValue::String(value) => {
                    for piece in value.iter() {
                        match piece {
                            StringPiece::Literal(piece) => {
                                self.push_str(piece);
                            }
                            StringPiece::Substitute(piece) => {
                                self.pieces.push(StringPiece::Substitute(piece.clone()));
                            }
                        }
                    }
                }
                MixedCompoundValue::Range(value) => match value {
                    RangeValue::StartEnd { start, end } => {
                        if let Some(start) = start {
                            self.push_value(elab, &Value::from(start.clone()));
                        }
                        match end {
                            RangeEnd::Exclusive(end) => {
                                self.push_str("..");
                                if let Some(end) = end {
                                    self.push_value(elab, &Value::from(end.clone()));
                                }
                            }
                            RangeEnd::Inclusive(end) => {
                                self.push_str("..=");
                                self.push_value(elab, &Value::from(end.clone()));
                            }
                        }
                    }
                    RangeValue::StartLength { start, length } => {
                        self.push_value(elab, &Value::from(start.clone()));
                        self.push_str("+..");
                        self.push_value(elab, &Value::from(length.clone()));
                    }
                },
                MixedCompoundValue::Tuple(elements) => {
                    self.push_str("(");
                    for (elem, last) in elements.iter().with_last() {
                        self.push_value(elab, elem);
                        if !last {
                            self.push_str(", ");
                        }
                    }
                    if elements.len() == 1 {
                        self.push_str(",");
                    }
                    self.push_str(")");
                }
                MixedCompoundValue::Struct(value) => {
                    let &StructValue { ty, ref fields } = value;
                    let ty_info = elab.struct_info(ty);

                    self.push_str(&ty_info.name);
                    self.push_str(".new(");
                    for ((field_name, field_value), last) in zip_eq(ty_info.fields.keys(), fields).with_last() {
                        self.push_str(field_name);
                        self.push_str("=");
                        self.push_value(elab, field_value);
                        if !last {
                            self.push_str(", ");
                        }
                    }
                    self.push_str(")");
                }
                MixedCompoundValue::Enum(value) => {
                    let &EnumValue {
                        ty,
                        variant,
                        ref payload,
                    } = value;
                    let ty_info = elab.enum_info(ty);

                    let (variant_name, _) = ty_info.variants.get_index(variant).unwrap();

                    self.push_str(&ty_info.name);
                    self.push_str(".");
                    self.push_str(variant_name);
                    if let Some(payload) = payload {
                        self.push_str("(");
                        self.push_value(elab, payload);
                        self.push_str(")");
                    }
                }
            },
            Value::Hardware(value) => {
                self.pieces.push(StringPiece::Substitute(value.clone()));
            }
        }
    }

    pub fn finish(self) -> Vec<StringPiece<String, HardwareValue>> {
        self.pieces
    }
}

impl CompileItemContext<'_, '_> {
    pub fn eval_string_literal(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        pieces: &[StringPiece<Span, Expression>],
    ) -> DiagResult<Value> {
        let elab = &self.refs.shared.elaboration_arenas;
        let mut builder = StringBuilder::new();

        let mut any_err = Ok(());
        for &piece in pieces {
            match piece {
                StringPiece::Literal(piece_span) => {
                    let raw = self.refs.fixed.source.span_str(piece_span);
                    let escaped = apply_string_literal_escapes(raw);
                    builder.push_str(escaped);
                }
                StringPiece::Substitute(piece_value) => {
                    let piece_value = self.eval_expression(scope, flow, &Type::Any, piece_value);
                    let piece_value = match piece_value {
                        Ok(v) => v,
                        Err(e) => {
                            any_err = Err(e);
                            continue;
                        }
                    };

                    builder.push_value(elab, &piece_value.inner);
                }
            }
        }

        any_err?;
        Ok(Value::Compound(MixedCompoundValue::String(Arc::new(builder.finish()))))
    }
}

pub fn hardware_print_string(
    elab: &ElaborationArenas,
    flow: &mut FlowHardware,
    large: &mut IrLargeArena,
    span: Span,
    pieces: &[StringPiece<String, HardwareValue>],
) {
    let mut new_ir_var = |info| flow.new_ir_variable(info);
    let mut builder = IrStringBuilder { span, pieces: vec![] };
    let mut block = vec![];

    for piece in pieces {
        match piece {
            StringPiece::Literal(s) => builder.push_str(s),
            StringPiece::Substitute(sub) => {
                block.extend(print_hardware_sub(elab, &mut new_ir_var, large, &mut builder, sub));
            }
        }
    }

    // flush any remaining pieces
    builder.print_and_clear(&mut block);

    // try to emit inline
    match block.len() {
        0 => {}
        1 => block.into_iter().for_each(|stmt| flow.push_ir_statement(stmt)),
        _ => flow.push_ir_statement(Spanned::new(span, IrStatement::Block(IrBlock { statements: block }))),
    }
}

struct IrStringBuilder {
    span: Span,
    pieces: Vec<IrStringPiece>,
}

impl IrStringBuilder {
    pub fn push_str(&mut self, s: &str) {
        if let Some(IrStringPiece::Literal(last)) = self.pieces.last_mut() {
            last.push_str(s);
        } else {
            self.pieces.push(IrStringPiece::Literal(s.to_owned()));
        }
    }

    pub fn push_sub(&mut self, sub: IrStringSubstitution) {
        self.pieces.push(IrStringPiece::Substitute(sub));
    }

    fn print_and_clear(&mut self, block: &mut Vec<Spanned<IrStatement>>) {
        if !self.pieces.is_empty() {
            let pieces = std::mem::take(&mut self.pieces);
            let stmt = IrStatement::Print(pieces);
            block.push(Spanned::new(self.span, stmt));
        }
    }
}

#[must_use]
fn print_hardware_sub(
    elab: &ElaborationArenas,
    new_ir_var: &mut impl FnMut(IrVariableInfo) -> IrVariable,
    large: &mut IrLargeArena,
    builder: &mut IrStringBuilder,
    value: &HardwareValue,
) -> Vec<Spanned<IrStatement>> {
    let span = builder.span;
    let mut block_parent = vec![];

    match &value.ty {
        HardwareType::Undefined => {
            builder.push_str("undef");
        }
        HardwareType::Bool => {
            // sadly we can't directly print booleans in verilog, so we need to do a full branch
            //   we could generate some common utility function for this, but that complicates the IR
            builder.print_and_clear(&mut block_parent);

            let block_print = |s: &str| IrBlock {
                statements: vec![Spanned::new(
                    span,
                    IrStatement::Print(vec![StringPiece::Literal(s.to_owned())]),
                )],
            };
            let stmt = IrIfStatement {
                condition: value.expr.clone(),
                then_block: block_print("true"),
                else_block: Some(block_print("false")),
            };
            block_parent.push(Spanned::new(span, IrStatement::If(stmt)));
        }
        HardwareType::Int(_) => {
            let sub = IrStringSubstitution::Integer(value.expr.clone(), IrIntegerRadix::Decimal);
            builder.push_sub(sub);
        }
        HardwareType::Tuple(element_types) => {
            builder.push_str("(");
            for (index, ty) in enumerate(element_types.as_ref()) {
                let element_expr = large.push_expr(IrExpressionLarge::TupleIndex {
                    base: value.expr.clone(),
                    index: BigUint::from(index),
                });
                let element_value = HardwareValue {
                    ty: ty.clone(),
                    domain: value.domain,
                    expr: element_expr,
                };
                block_parent.extend(print_hardware_sub(elab, new_ir_var, large, builder, &element_value));
                if index < element_types.len() - 1 {
                    builder.push_str(", ");
                }
            }
            if element_types.len() == 1 {
                builder.push_str(",");
            }
            builder.push_str(")");
        }
        HardwareType::Array(ty_inner, len) => {
            builder.push_str("[");

            let mut element_value = |index: IrExpression| {
                let element_expr = large.push_expr(IrExpressionLarge::ArrayIndex {
                    base: value.expr.clone(),
                    index,
                });
                HardwareValue {
                    ty: ty_inner.as_ref().clone(),
                    domain: value.domain,
                    expr: element_expr,
                }
            };

            if len.is_zero() {
                // empty array
            } else if len == &BigUint::ONE {
                // single element, just print it
                // TODO also do this for short arrays with simplem elements?
                let element_value = element_value(IrExpression::Int(BigInt::ZERO));
                block_parent.extend(print_hardware_sub(elab, new_ir_var, large, builder, &element_value));
            } else {
                // multiple elements, emit a loop

                // print any pending pieces before the loop
                builder.print_and_clear(&mut block_parent);

                // create index variable
                let range = ClosedIncRange {
                    start_inc: BigInt::ZERO,
                    end_inc: len - 1,
                };
                let index = new_ir_var(IrVariableInfo {
                    ty: IrType::Int(range.clone()),
                    debug_info_span: span,
                    debug_info_id: Some("i_str".to_owned()),
                });

                // print the element, clearing again immediately afterwards to force all statements into the loop
                let element_value = element_value(IrExpression::Variable(index));
                let mut element_block = print_hardware_sub(elab, new_ir_var, large, builder, &element_value);
                builder.print_and_clear(&mut element_block);

                // print comma
                let if_cond = IrExpressionLarge::IntCompare(
                    IrIntCompareOp::Lt,
                    IrExpression::Variable(index),
                    IrExpression::Int(range.end_inc.clone()),
                );
                let if_stmt = IrIfStatement {
                    condition: large.push_expr(if_cond),
                    then_block: IrBlock {
                        statements: vec![Spanned::new(
                            span,
                            IrStatement::Print(vec![StringPiece::Literal(", ".to_owned())]),
                        )],
                    },
                    else_block: None,
                };
                element_block.push(Spanned::new(span, IrStatement::If(if_stmt)));

                // push the loop
                let for_stmt = IrForStatement {
                    index,
                    range,
                    block: IrBlock {
                        statements: element_block,
                    },
                };
                block_parent.push(Spanned::new(span, IrStatement::For(for_stmt)));
            }
            builder.push_str("]");
        }
        HardwareType::Struct(ty) => {
            let ty_info = elab.struct_info(ty.inner());
            let ty_fields_hw = ty_info.fields_hw.as_ref().unwrap();

            builder.push_str(&ty_info.name);
            builder.push_str(".new(");
            for (field_index, ((field_name, _), field_ty)) in enumerate(zip_eq(&ty_info.fields, ty_fields_hw)) {
                let field_expr = large.push_expr(IrExpressionLarge::TupleIndex {
                    base: value.expr.clone(),
                    index: BigUint::from(field_index),
                });
                let field_value = HardwareValue {
                    ty: field_ty.clone(),
                    domain: value.domain,
                    expr: field_expr,
                };

                builder.push_str(field_name);
                builder.push_str("=");
                block_parent.extend(print_hardware_sub(elab, new_ir_var, large, builder, &field_value));

                if field_index < ty_info.fields.len() - 1 {
                    builder.push_str(", ");
                }
            }
            builder.push_str(")");
        }
        HardwareType::Enum(ty) => {
            let ty_info = elab.enum_info(ty.inner());
            let ty_info_hw = ty_info.hw.as_ref().unwrap();

            builder.push_str(&ty_info.name);
            builder.push_str(".");
            builder.print_and_clear(&mut block_parent);

            let mut rest = IrStatement::Print(vec![IrStringPiece::Literal("<unknown>".to_owned())]);
            for (variant, (variant_name, _)) in enumerate(&ty_info.variants).rev() {
                let cond = ty_info_hw.check_tag_matches(large, value.expr.clone(), variant);
                let payload = ty_info_hw.extract_payload(large, value, variant);

                let mut block_case = vec![];
                builder.push_str(variant_name);
                if let Some(payload) = payload {
                    builder.push_str("(");
                    block_case.extend(print_hardware_sub(elab, new_ir_var, large, builder, &payload));
                    builder.push_str(")");
                    builder.print_and_clear(&mut block_case);
                }

                rest = IrStatement::If(IrIfStatement {
                    condition: cond,
                    then_block: IrBlock { statements: block_case },
                    else_block: Some(IrBlock::new_single(span, rest)),
                })
            }

            block_parent.push(Spanned::new(span, rest));
        }
    }

    block_parent
}

impl CompileValue {
    pub fn value_string(&self, elab: &ElaborationArenas) -> String {
        match self {
            CompileValue::Simple(v) => v.value_string(elab),
            CompileValue::Compound(v) => v.value_string(elab),
            CompileValue::Hardware(never) => never.unreachable(),
        }
    }
}

impl SimpleCompileValue {
    pub fn value_string(&self, elab: &ElaborationArenas) -> String {
        match self {
            SimpleCompileValue::Type(v) => v.value_string(elab),
            SimpleCompileValue::Bool(v) => v.to_string(),
            SimpleCompileValue::Int(v) => v.to_string(),
            SimpleCompileValue::Array(v) => {
                let content = v.iter().map(|e| e.value_string(elab)).format(", ");
                format!("[{}]", content)
            }
            // TODO include names
            SimpleCompileValue::Function(_) => "<function>".to_owned(),
            SimpleCompileValue::Module(_) => "<module>".to_owned(),
            SimpleCompileValue::Interface(_) => "<interface>".to_owned(),
            SimpleCompileValue::InterfaceView(_) => "<interface view>".to_owned(),
        }
    }
}

impl Type {
    pub fn value_string(&self, elab: &ElaborationArenas) -> String {
        match self {
            Type::Type => "type".to_string(),
            Type::Any => "any".to_string(),
            Type::Undefined => "undefined".to_string(),

            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Int(range) => {
                let range_int = IncRange::OPEN;
                let range_uint = IncRange {
                    start_inc: Some(BigInt::ZERO),
                    end_inc: None,
                };
                if range == &range_int {
                    "int".to_owned()
                } else if range == &range_uint {
                    "uint".to_owned()
                } else {
                    format!("int({range})")
                }
            }
            Type::Tuple(inner) => {
                let mut f = String::new();
                swrite!(f, "(");
                swrite!(f, "{}", inner.iter().map(|e| e.value_string(elab)).format(", "));
                if inner.len() == 1 {
                    swrite!(f, ",");
                }
                swrite!(f, ")");
                f
            }
            Type::Array(inner, len) => {
                format!("[{len}]{}", inner.value_string(elab))
            }
            Type::Struct(ty) => {
                let ty_info = elab.struct_info(*ty);
                format!("<struct {}>", ty_info.name)
            }
            Type::Enum(ty) => {
                let ty_info = elab.enum_info(*ty);
                format!("<enum {}>", ty_info.name)
            }
            Type::Range => "<range>".to_string(),
            // TODO include names
            Type::Function => "<function>".to_string(),
            Type::Module => "<module>".to_string(),
            Type::Interface => "<interface>".to_string(),
            Type::InterfaceView => "<interface_view>".to_string(),
            Type::Builtin => format!("<{TOKEN_STR_BUILTIN}>"),
        }
    }
}

impl HardwareType {
    pub fn value_string(&self, elab: &ElaborationArenas) -> String {
        self.as_type().value_string(elab)
    }
}

impl CompileCompoundValue {
    pub fn value_string(&self, elab: &ElaborationArenas) -> String {
        match self {
            CompileCompoundValue::String(v) => v.as_ref().clone(),
            CompileCompoundValue::Range(v) => {
                let IncRange { start_inc, end_inc } = v;

                let mut f = String::new();
                if let Some(start) = start_inc {
                    swrite!(f, "{start}");
                }
                swrite!(f, "..");
                if let Some(end) = end_inc {
                    swrite!(f, "{end}");
                }
                f
            }
            CompileCompoundValue::Tuple(v) => {
                let mut f = String::new();
                swrite!(f, "(");
                swrite!(f, "{}", v.iter().map(|e| e.value_string(elab)).format(", "));
                if v.len() == 1 {
                    swrite!(f, ",");
                }
                swrite!(f, ")");
                f
            }
            CompileCompoundValue::Struct(v) => {
                let &StructValue { ty, ref fields } = v;
                let ty_info = elab.struct_info(ty);

                let mut f = String::new();
                swrite!(f, "{}.new(", ty_info.name);
                for ((field_name, field_value), last) in zip_eq(ty_info.fields.keys(), fields).with_last() {
                    swrite!(f, "{}={}", field_name, field_value.value_string(elab));
                    if !last {
                        swrite!(f, ", ");
                    }
                }
                swrite!(f, ")");
                f
            }
            CompileCompoundValue::Enum(v) => {
                let &EnumValue {
                    ty,
                    variant,
                    ref payload,
                } = v;
                let ty_info = elab.enum_info(ty);

                let (variant_name, _) = ty_info.variants.get_index(variant).unwrap();

                let mut f = String::new();
                swrite!(f, "{}.{}", ty_info.name, variant_name);
                if let Some(payload) = payload {
                    swrite!(f, "({})", payload.value_string(elab));
                }
                f
            }
        }
    }
}
