use crate::data::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::data::parsed::AstRefItem;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::CompileState;
use crate::new::misc::{DomainSignal, ScopedEntry, TypeOrValue, TypeOrValueNoError};
use crate::new::types::Type;
use crate::new::value::{CompileValue, ExpressionValue, ScopedValue};
use crate::syntax::ast;
use crate::syntax::ast::{DomainKind, Expression, ExpressionKind, IntPattern, Item, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use itertools::Itertools;
use num_bigint::BigInt;

impl CompileState<'_> {
    pub fn eval_expression_as_any(&mut self, scope: Scope, expr: &Expression) -> TypeOrValue<ExpressionValue> {
        match &expr.inner {
            ExpressionKind::Dummy =>
                TypeOrValue::Error(self.diags.report_todo(expr.span, "expr kind Dummy")),
            ExpressionKind::Wrapped(inner) =>
                self.eval_expression_as_any(scope, inner),
            ExpressionKind::Id(id) => {
                match self.scopes[scope].find(&self.scopes, self.diags, id, Visibility::Private) {
                    Ok(found) => match found.value {
                        &ScopedEntry::Item(item) => self.eval_item_as_any(item),
                        ScopedEntry::Direct(direct) => match direct {
                            TypeOrValue::Type(ty) => TypeOrValue::Type(ty.clone()),
                            TypeOrValue::Value(v) => match v {
                                ScopedValue::Compile(v) => TypeOrValue::Value(ExpressionValue::Compile(v.clone())),
                                &ScopedValue::Port(v) => TypeOrValue::Value(ExpressionValue::Port(v)),
                                &ScopedValue::Wire(v) => TypeOrValue::Value(ExpressionValue::Wire(v)),
                                &ScopedValue::Register(v) => TypeOrValue::Value(ExpressionValue::Register(v)),
                                &ScopedValue::Variable(v) => TypeOrValue::Value(ExpressionValue::Variable(v)),
                            },
                            &TypeOrValue::Error(e) => TypeOrValue::Error(e),
                        },
                    },
                    Err(e) => TypeOrValue::Error(e),
                }
            }

            ExpressionKind::TypeFunc(_, _) => self.diags.report_todo(expr.span, "expr kind TypeFunc").into(),
            ExpressionKind::IntPattern(ref pattern) =>
                match pattern {
                    IntPattern::Hex(_) =>
                        TypeOrValue::Error(self.diags.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        TypeOrValue::Error(self.diags.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(pattern_raw) => {
                        let pattern_clean = pattern_raw.replace("_", "");
                        match pattern_clean.parse::<BigInt>() {
                            Ok(value) => TypeOrValue::Value(ExpressionValue::Compile(CompileValue::Int(value))),
                            Err(_) => self.diags.report_internal_error(expr.span, "failed to parse int-pattern").into(),
                        }
                    }
                }
            &ExpressionKind::BoolLiteral(literal) =>
                TypeOrValue::Value(ExpressionValue::Compile(CompileValue::Bool(literal))),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) =>
                TypeOrValue::Value(ExpressionValue::Compile(CompileValue::String(literal.clone()))),

            ExpressionKind::ArrayLiteral(_) => self.diags.report_todo(expr.span, "expr kind ArrayLiteral").into(),
            ExpressionKind::TupleLiteral(_) => self.diags.report_todo(expr.span, "expr kind TupleLiteral").into(),
            ExpressionKind::StructLiteral(_) => self.diags.report_todo(expr.span, "expr kind StructLiteral").into(),
            ExpressionKind::RangeLiteral(_) => self.diags.report_todo(expr.span, "expr kind RangeLiteral").into(),
            ExpressionKind::UnaryOp(_, _) => self.diags.report_todo(expr.span, "expr kind UnaryOp").into(),
            ExpressionKind::BinaryOp(_, _, _) => self.diags.report_todo(expr.span, "expr kind BinaryOp").into(),
            ExpressionKind::TernarySelect(_, _, _) => self.diags.report_todo(expr.span, "expr kind TernarySelect").into(),
            ExpressionKind::ArrayIndex(_, _) => self.diags.report_todo(expr.span, "expr kind ArrayIndex").into(),
            ExpressionKind::DotIdIndex(_, _) => self.diags.report_todo(expr.span, "expr kind DotIdIndex").into(),
            ExpressionKind::DotIntIndex(_, _) => self.diags.report_todo(expr.span, "expr kind DotIntIndex").into(),
            ExpressionKind::Call(_, _) => self.diags.report_todo(expr.span, "expr kind Call").into(),
            ExpressionKind::Builtin(ref args) => self.eval_builtin(scope, expr.span, args),
        }
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin(&mut self, scope: Scope, expr_span: Span, args: &Spanned<Vec<Expression>>) -> TypeOrValue<ExpressionValue> {
        let mut args_eval = Ok(vec![]);

        // evaluate args
        for arg in &args.inner {
            let eval = match self.eval_expression_as_any(scope, arg) {
                TypeOrValue::Type(ty) => TypeOrValue::Type(ty),
                TypeOrValue::Value(v) => {
                    let err_value = |s| {
                        self.diags.report_simple("__builtin arguments must be compile-time constants", arg.span, format!("got {}", s))
                    };

                    match v {
                        ExpressionValue::Error(e) => TypeOrValue::Error(e),
                        ExpressionValue::Compile(v) => TypeOrValue::Value(v),
                        ExpressionValue::Port(_) => err_value("port").into(),
                        ExpressionValue::Wire(_) => err_value("wire").into(),
                        ExpressionValue::Register(_) => err_value("register").into(),
                        ExpressionValue::Variable(_) => err_value("variable").into(),
                        ExpressionValue::RuntimeExpression { .. } => err_value("runtime expression").into(),
                    }
                }
                TypeOrValue::Error(e) => TypeOrValue::Error(e),
            };

            match (&mut args_eval, eval) {
                (Ok(args_eval), TypeOrValue::Type(ty)) => args_eval.push(TypeOrValueNoError::Type(ty)),
                (Ok(args_eval), TypeOrValue::Value(v)) => args_eval.push(TypeOrValueNoError::Value(v)),
                (&mut Err(e), _) | (_, TypeOrValue::Error(e)) => args_eval = Err(e),
            }
        }
        let args_eval = match args_eval {
            Ok(args_eval) => args_eval,
            Err(e) => return TypeOrValue::Error(e),
        };

        if let (Some(TypeOrValueNoError::Value(CompileValue::String(a0))), Some(TypeOrValueNoError::Value(CompileValue::String(a1)))) = (args_eval.get(0), args_eval.get(1)) {
            let rest = &args_eval[2..];
            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return TypeOrValue::Type(Type::Any),
                ("type", "bool", []) => return TypeOrValue::Type(Type::Bool),
                ("type", "int_range", [range]) => {
                    if let TypeOrValueNoError::Value(CompileValue::IntRange(range)) = range {
                        return TypeOrValue::Type(Type::Int(range.clone()));
                    }
                }

                ("value", "undefined", []) => return TypeOrValue::Value(ExpressionValue::Compile(CompileValue::Undefined)),

                // fallthrough into err
                _ => {}
            }
        }

        let diag = Diagnostic::new("invalid __builtin args")
            .snippet(expr_span)
            .add_error(args.span, "invalid args")
            .finish()
            .finish();
        TypeOrValue::Error(self.diags.report(diag))
    }

    fn eval_item_as_any(&mut self, item: AstRefItem) -> TypeOrValue<ExpressionValue> {
        let file_scope = match self.file_scopes.get(&item.file()).unwrap() {
            Ok(file_scopes) => file_scopes.scope_inner_import,
            &Err(e) => return TypeOrValue::Error(e),
        };

        match &self.parsed[item] {
            Item::Import(item) => self.diags.report_internal_error(item.span, "imports should have been resolved already").into(),
            Item::Const(item) => self.diags.report_todo(item.span, "eval item kind Const").into(),
            Item::Type(item) => {
                let ast::ItemDefType { span: _, vis: _, id: _, params, inner } = item;

                match params {
                    None => TypeOrValue::Type(self.eval_expression_as_ty(file_scope, inner)),
                    Some(_) => self.diags.report_todo(item.span, "type with parameters").into(),
                }
            }
            Item::Struct(item) => self.diags.report_todo(item.span, "eval item kind Struct").into(),
            Item::Enum(item) => self.diags.report_todo(item.span, "eval item kind Enum").into(),
            Item::Function(item) => self.diags.report_todo(item.span, "eval item kind Function").into(),
            Item::Module(item) => self.diags.report_todo(item.span, "eval item kind Module").into(),
            Item::Interface(item) => self.diags.report_todo(item.span, "eval item kind Interface").into(),
        }
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> Type {
        match self.eval_expression_as_any(scope, expr) {
            TypeOrValue::Type(ty) => ty,
            TypeOrValue::Value(_) => {
                let e = self.diags.report_simple("expected type, got value", expr.span, "got value");
                Type::Error(e)
            }
            TypeOrValue::Error(e) => Type::Error(e),
        }
    }

    pub fn eval_expression_as_value_any(&mut self, scope: Scope, expr: &Expression) -> ExpressionValue {
        let _ = (scope, expr);
        todo!()
    }

    // TODO enforce domain, here or more inward?
    pub fn eval_expression_as_value_read(&mut self, scope: Scope, expr: &Expression) -> ExpressionValue {
        let _ = (scope, expr);
        todo!()
    }

    pub fn eval_expression_as_value_read_compile(&mut self, scope: Scope, expr: &Expression) -> ExpressionValue {
        let _ = (scope, expr);
        todo!()
    }

    pub fn eval_expression_as_domain_signal_read(&mut self, scope: Scope, expr: &Expression) -> DomainSignal {
        let _ = (scope, expr);
        todo!()
    }

    pub fn eval_domain(&mut self, scope: Scope, domain: &DomainKind<Box<Expression>>) -> DomainKind<DomainSignal> {
        match domain {
            DomainKind::Async => DomainKind::Async,
            DomainKind::Sync(domain) => {
                DomainKind::Sync(SyncDomain {
                    clock: self.eval_expression_as_domain_signal_read(scope, &domain.clock),
                    reset: self.eval_expression_as_domain_signal_read(scope, &domain.reset),
                })
            }
        }
    }
}