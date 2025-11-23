use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::DiagResult;
use crate::front::flow::Flow;
use crate::front::scope::Scope;
use crate::front::types::Type;
use crate::front::value::{CompoundValue, HardwareValue, Value, ValueCommon};
use crate::syntax::ast::{Expression, StringPiece};
use crate::syntax::pos::Span;
use crate::syntax::token::apply_string_literal_escapes;
use std::borrow::Cow;
use std::sync::Arc;

struct StringBuilder {
    pieces: Vec<StringPiece<String, HardwareValue>>,
}

impl StringBuilder {
    pub fn new() -> Self {
        Self { pieces: vec![] }
    }

    pub fn push(&mut self, next: StringPiece<Cow<str>, HardwareValue>) {
        match (self.pieces.last_mut(), next) {
            // concatenate adjacent literals to canonicalize
            (Some(StringPiece::Literal(last)), StringPiece::Literal(next)) => {
                last.push_str(&next);
            }
            // just push the new piece
            (_, StringPiece::Literal(next)) => self.pieces.push(StringPiece::Literal(next.into_owned())),
            (_, StringPiece::Substitute(next)) => self.pieces.push(StringPiece::Substitute(next)),
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
        let mut builder = StringBuilder::new();

        let mut any_err = Ok(());
        for &piece in pieces {
            match piece {
                StringPiece::Literal(piece_span) => {
                    let raw = self.refs.fixed.source.span_str(piece_span);
                    let escaped = apply_string_literal_escapes(raw);
                    builder.push(StringPiece::Literal(escaped));
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

                    match piece_value.inner {
                        Value::Simple(v) => {
                            builder.push(StringPiece::Literal(Cow::Owned(v.diagnostic_string())));
                        }
                        Value::Compound(v) => match v {
                            CompoundValue::String(s) => {
                                for p in Arc::unwrap_or_clone(s) {
                                    let p = match p {
                                        StringPiece::Literal(p) => StringPiece::Literal(Cow::Owned(p)),
                                        StringPiece::Substitute(p) => StringPiece::Substitute(p),
                                    };
                                    builder.push(p);
                                }
                            }
                            // TODO proper string formatting
                            // TODO avoid code duplication between
                            //   * this (for compile-time string formatting)
                            //   * diagnostic_str (for diagnostic string formatting)
                            //   * print (where hardware values are actually printed)
                            CompoundValue::Range(_) => builder.push(StringPiece::Literal(Cow::Borrowed("range"))),
                            CompoundValue::Tuple(_) => builder.push(StringPiece::Literal(Cow::Borrowed("tuple"))),
                            CompoundValue::Struct(_) => builder.push(StringPiece::Literal(Cow::Borrowed("struct"))),
                            CompoundValue::Enum(_) => builder.push(StringPiece::Literal(Cow::Borrowed("enum"))),
                        },
                        Value::Hardware(v) => {
                            // TODO descend until we hit simple hardware values here already?
                            //   or should we move that logic into print?
                            builder.push(StringPiece::Substitute(v));
                        }
                    }
                }
            }
        }

        any_err?;
        Ok(Value::Compound(CompoundValue::String(Arc::new(builder.finish()))))
    }
}
