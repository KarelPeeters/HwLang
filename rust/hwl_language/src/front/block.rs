use crate::data::compiled::VariableInfo;
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::module_body::LowerStatement;
use crate::front::checking::DomainUserControlled;
use crate::front::common::{ContextDomainKind, ExpressionContext, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::Type;
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{Block, BlockStatement, BlockStatementKind, PortDirection, Spanned, VariableDeclaration};
use crate::syntax::pos::Span;

#[derive(Debug, Copy, Clone)]
pub enum AccessDirection {
    Read,
    Write,
}

impl CompileState<'_, '_> {
    #[must_use]
    pub fn visit_block(
        &mut self,
        ctx: &ExpressionContext,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> Vec<LowerStatement> {
        let diags = self.diags;
        let scope = self.compiled.scopes.new_child(parent_scope, block.span, Visibility::Private);

        let mut result_statements = vec![];

        for statement in &block.statements {
            match &statement.inner {
                BlockStatementKind::ConstDeclaration(decl) => {
                    self.process_and_declare_const(scope, decl, Visibility::Private);
                }
                BlockStatementKind::VariableDeclaration(decl) => {
                    let VariableDeclaration { span: _, mutable, id, ty, init } = decl;
                    let mutable = *mutable;

                    let ty_eval = ty.as_ref().map(|ty| {
                        let inner = self.eval_expression_as_ty(scope, ty);
                        Spanned { span: ty.span, inner }
                    });
                    let init_unchecked = self.eval_expression_as_value(ctx, scope, init);

                    // check or infer type
                    let (ty_eval, init_eval) = match ty_eval {
                        None => (self.type_of_value(init.span, &init_unchecked), init_unchecked),
                        Some(ty_eval) => {
                            match self.check_type_contains(Some(ty_eval.span), init.span, &ty_eval.inner, &init_unchecked) {
                                Ok(()) => (ty_eval.inner, init_unchecked),
                                Err(e) => (Type::Error(e), Value::Error(e)),
                            }
                        }
                    };

                    // check domain
                    match ctx.domain() {
                        ContextDomainKind::Specific(domain) => {
                            let init_domain = self.domain_of_value(init.span, &init_eval);
                            let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
                                decl.span,
                                &domain,
                                init.span,
                                &init_domain,
                                DomainUserControlled::Source,
                                "variable initializer must be assignable to context domain",
                            );
                        }
                        ContextDomainKind::Passthrough => {}
                    }

                    // declare
                    let info = VariableInfo { defining_id: id.clone(), ty: ty_eval.clone(), mutable };
                    let variable = self.compiled.variables.push(info);
                    let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(variable))));
                    self.compiled[scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(assignment) => {
                    let &ast::Assignment { span: _, op, ref target, ref value } = assignment;

                    let target_eval = self.eval_expression_as_value(ctx, scope, target);
                    let value_eval = self.eval_expression_as_value(ctx, scope, value);

                    let mut any_err = Ok(());

                    // check type
                    let target_ty = self.type_of_value(target.span, &target_eval);
                    any_err = any_err.and(self.check_type_contains(Some(target.span), value.span, &target_ty, &value_eval));

                    // check read_write
                    any_err = any_err.and(self.check_value_usable_as_direction(ctx, target.span, &target_eval, AccessDirection::Write));
                    any_err = any_err.and(self.check_value_usable_as_direction(ctx, value.span, &value_eval, AccessDirection::Read));

                    // check domain
                    let target_domain = self.domain_of_value(target.span, &target_eval);
                    let value_domain = self.domain_of_value(value.span, &value_eval);

                    match ctx.domain() {
                        ContextDomainKind::Passthrough => {
                            // target/value need to be compatible, without additional constraints
                            any_err = any_err.and(self.check_domain_crossing(
                                target.span,
                                &target_domain,
                                value.span,
                                &value_domain,
                                DomainUserControlled::Both,
                                "assignment target domain must be assignable from value domain",
                            ))
                        }
                        ContextDomainKind::Specific(domain) => {
                            // target/domain and domain/value both need to be compatible
                            any_err = any_err.and(self.check_domain_crossing(
                                target.span,
                                &target_domain,
                                ctx.span(),
                                &domain,
                                DomainUserControlled::Target,
                                "assignment target domain must be assignable from context domain",
                            ));
                            any_err = any_err.and(self.check_domain_crossing(
                                ctx.span(),
                                &domain,
                                value.span,
                                &value_domain,
                                DomainUserControlled::Source,
                                "assignment context domain must be assignable from value domain",
                            ));
                        }
                    };

                    // operator assignments are not implemented yet
                    if op.inner.is_some() {
                        any_err = Err(diags.report_todo(assignment.span, "assignment with operator"));
                    }

                    // result
                    let statement = match any_err {
                        Ok(()) => {
                            LowerStatement::Assignment {
                                target: Spanned { span: target.span, inner: target_eval },
                                value: Spanned { span: value.span, inner: value_eval },
                            }
                        }
                        Err(e) => LowerStatement::Error(e),
                    };
                    result_statements.push(statement);
                }
                BlockStatementKind::Expression(expression) => {
                    match ctx {
                        ExpressionContext::FunctionBody { .. } => {
                            // TODO control flow reachability and return checking
                            let _ = self.eval_expression_as_value(ctx, scope, expression);
                        }
                        _ => {
                            diags.report_todo(statement.span, "expression statement outside of function body");
                        }
                    }
                }
            }
        }

        result_statements
    }

    // TODO this could be much faster with proper LRValue-style tracking
    pub fn check_value_usable_as_direction(&self, ctx: &ExpressionContext, value_span: Span, value: &Value, dir: AccessDirection) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;

        const READ_LABEL: &str = "reading here";
        const WRITE_LABEL: &str = "writing here";

        let if_write_simple_error = |kind: &str| {
            match dir {
                AccessDirection::Read => Ok(()),
                AccessDirection::Write => {
                    let e = diags.report_simple(
                        format!("invalid write: {kind} is never writable"),
                        value_span,
                        WRITE_LABEL,
                    );
                    Err(e)
                }
            }
        };
        let if_write_simple_error_with_location = |kind: &str, def_span: Span| {
            match dir {
                AccessDirection::Read => Ok(()),
                AccessDirection::Write => {
                    let diag = Diagnostic::new(format!("invalid write: {kind} is never writable"))
                        .add_error(value_span, WRITE_LABEL)
                        .add_info(def_span, format!("{kind} declared here"))
                        .finish();
                    Err(diags.report(diag))
                }
            }
        };

        match value {
            // propagate error
            &Value::Error(e) => Err(e),

            // maybe assignable, depending on the details and context
            &Value::ModulePort(port) => {
                match (dir, self.compiled[port].direction) {
                    (AccessDirection::Read, PortDirection::Input) => Ok(()),
                    (AccessDirection::Write, PortDirection::Output) => Ok(()),
                    (AccessDirection::Read, PortDirection::Output) =>
                        Err(diags.report_simple("invalid read: cannot read from output port", value_span, READ_LABEL)),
                    (AccessDirection::Write, PortDirection::Input) =>
                        Err(diags.report_simple("invalid write: cannot write to input port", value_span, WRITE_LABEL)),
                }
            }
            &Value::Wire(wire) => {
                let info = &self.compiled[wire];

                match (dir, ctx) {
                    (AccessDirection::Read, ExpressionContext::CombinatorialBlock(_) | ExpressionContext::ClockedBlock(_)) =>
                        Ok(()),
                    (AccessDirection::Read, _) =>
                        Err(diags.report_simple("invalid read: wire can only be read inside combinatorial or clocked block", value_span, READ_LABEL)),
                    (AccessDirection::Write, ExpressionContext::CombinatorialBlock(_) | ExpressionContext::ClockedBlock(_)) => {
                        if info.has_declaration_value {
                            Err(diags.report_simple("invalid write: wire already has declaration value", value_span, WRITE_LABEL))
                        } else {
                            Ok(())
                        }
                    }
                    (AccessDirection::Write, _) =>
                        Err(diags.report_simple("invalid write: wire can only be written inside combinatorial or clocked block", value_span, WRITE_LABEL)),
                }
            }
            &Value::Register(_) => {
                match (dir, ctx) {
                    (AccessDirection::Read, ExpressionContext::ClockedBlock(_) | ExpressionContext::CombinatorialBlock(_)) =>
                        Ok(()),
                    (AccessDirection::Read, _) =>
                        Err(diags.report_simple("invalid read: register can only be read inside clocked or combinatorial block", value_span, READ_LABEL)),
                    (AccessDirection::Write, ExpressionContext::ClockedBlock(_)) =>
                        Ok(()),
                    (AccessDirection::Write, _) =>
                        Err(diags.report_simple("invalid write: register can only be written inside clocked block", value_span, WRITE_LABEL)),
                }
            }
            Value::Variable(var) => {
                match dir {
                    AccessDirection::Read => Ok(()),
                    AccessDirection::Write => {
                        let info = &self.compiled.variables[*var];
                        if info.mutable {
                            Ok(())
                        } else {
                            let diag = Diagnostic::new("invalid write: variable is not mutable")
                                .add_error(value_span, WRITE_LABEL)
                                .add_info(info.defining_id.span(), "variable declared here")
                                .finish();
                            Err(diags.report(diag))
                        }
                    }
                }
            }

            // only readable assignable
            Value::Never => if_write_simple_error("_never_ value"),
            Value::Unit => if_write_simple_error("unit value"),
            Value::BoolConstant(_) => if_write_simple_error("boolean constant"),
            Value::IntConstant(_) => if_write_simple_error("integer constant value"),
            Value::StringConstant(_) => if_write_simple_error("string constant value"),
            Value::Module(_) => if_write_simple_error("module value"),

            // TODO function call args should be checked for read/write
            // TODO re-implement all of this as a LRValue system, that checks for read/write
            //   once it's converted to the underlying value
            Value::FunctionReturn(_) => if_write_simple_error("function return value"),

            &Value::GenericParameter(param) =>
                if_write_simple_error_with_location("generic parameter", self.compiled[param].defining_id.span),
            &Value::Constant(cst) =>
                if_write_simple_error_with_location("constant", self.compiled[cst].defining_id.span()),

            // recursively check sub-expressions for read-ability
            Value::Range(info) => {
                let RangeInfo { start, end } = info;

                let mut r = Ok(());
                if let Some(start) = start {
                    r = r.and(self.check_value_usable_as_direction(ctx, value_span, start, AccessDirection::Read));
                }
                if let Some(end) = end {
                    r = r.and(self.check_value_usable_as_direction(ctx, value_span, end, AccessDirection::Read));
                }

                r = r.and(if_write_simple_error("range expression"));
                r
            }
            Value::Binary(_, left, right) => {
                let mut r = self.check_value_usable_as_direction(ctx, value_span, left, AccessDirection::Read);
                r = r.and(self.check_value_usable_as_direction(ctx, value_span, right, AccessDirection::Read));
                r = r.and(if_write_simple_error("binary expression"));
                r
            }
            Value::UnaryNot(x) => {
                let mut r = self.check_value_usable_as_direction(ctx, value_span, x, AccessDirection::Read);
                r = r.and(if_write_simple_error("unary not expression"));
                r
            }
        }
    }
}
