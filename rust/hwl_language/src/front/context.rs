use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::domain::{BlockDomain, DomainSignal, ValueDomain};
use crate::front::implication::Implication;
use crate::front::module::ExtraRegisterInit;
use crate::front::signal::Signal;
use crate::front::types::HardwareType;
use crate::front::value::MaybeUndefined;
use crate::front::variables::{ValueVersioned, VariableValues};
use crate::mid::ir::{
    IrBlock, IrExpression, IrRegister, IrRegisterInfo, IrRegisters, IrStatement, IrVariable, IrVariableInfo,
    IrVariables,
};
use crate::syntax::ast::{MaybeIdentifier, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::throw;
use annotate_snippets::Level;
use std::fmt::Debug;

pub trait ExpressionContext {
    type Block: Debug;

    // TODO rework IR unwrapping, look at what's actually necessary,
    //   maybe just `unwrap(&mut self) -> IrExpressionContext` is enough
    fn new_ir_block(&self) -> Self::Block;

    fn push_ir_statement(
        &self,
        diags: &Diagnostics,
        block: &mut Self::Block,
        stmt: Spanned<IrStatement>,
    ) -> Result<(), ErrorGuaranteed>;

    fn push_ir_statement_block(&self, block_parent: &mut Self::Block, block_inner: Spanned<Self::Block>);

    fn unwrap_ir_block(&self, diags: &Diagnostics, span: Span, block: Self::Block) -> Result<IrBlock, ErrorGuaranteed>;

    fn new_ir_variable(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        info: IrVariableInfo,
    ) -> Result<IrVariable, ErrorGuaranteed>;

    fn new_ir_register(
        &mut self,
        ctx: &CompileItemContext,
        diags: &Diagnostics,
        id: MaybeIdentifier,
        ty: HardwareType,
        init: Spanned<MaybeUndefined<IrExpression>>,
    ) -> Result<(IrRegister, SyncDomain<DomainSignal>), ErrorGuaranteed>;

    fn report_assignment(
        &mut self,
        ctx: &CompileItemContext,
        target: Spanned<Signal>,
        vars: &mut VariableValues,
    ) -> Result<(), ErrorGuaranteed>;

    fn block_domain(&self) -> BlockDomain;

    fn condition_domains(&self) -> &[Spanned<ValueDomain>];

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed>;

    fn with_implications<T>(
        &mut self,
        diags: &Diagnostics,
        implications: Vec<Implication>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed>;

    fn for_each_implication(&self, value: ValueVersioned, f: impl FnMut(&Implication));

    fn is_ir_context(&self) -> bool;

    fn check_ir_context(&self, diags: &Diagnostics, span: Span, reason: &str) -> Result<(), ErrorGuaranteed>;
}

pub struct CompileTimeExpressionContext {
    pub span: Span,
    pub reason: String,
}

// TODO some of these errors should probably be real errors, instead of internal compiler errors
impl ExpressionContext for CompileTimeExpressionContext {
    type Block = ();

    fn new_ir_block(&self) {}

    fn push_ir_statement(
        &self,
        diags: &Diagnostics,
        statements: &mut Self::Block,
        ir_statement: Spanned<IrStatement>,
    ) -> Result<(), ErrorGuaranteed> {
        let &mut () = statements;
        Err(diags.report_internal_error(ir_statement.span, "trying to push IR statement in compile-time context"))
    }

    fn push_ir_statement_block(&self, block_parent: &mut Self::Block, block_inner: Spanned<Self::Block>) {
        let &mut () = block_parent;
        let () = block_inner.inner;
        // do nothing
    }

    fn unwrap_ir_block(&self, diags: &Diagnostics, span: Span, block: Self::Block) -> Result<IrBlock, ErrorGuaranteed> {
        let () = block;
        Err(diags.report_internal_error(span, "trying to unwrap IR block in compile-time context"))
    }

    fn new_ir_variable(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        info: IrVariableInfo,
    ) -> Result<IrVariable, ErrorGuaranteed> {
        let _ = info;
        Err(diags.report_internal_error(span, "trying to create IR variable in compile-time context"))
    }

    fn new_ir_register(
        &mut self,
        ctx: &CompileItemContext,
        diags: &Diagnostics,
        id: MaybeIdentifier,
        ty: HardwareType,
        init: Spanned<MaybeUndefined<IrExpression>>,
    ) -> Result<(IrRegister, SyncDomain<DomainSignal>), ErrorGuaranteed> {
        let _ = (ctx, ty, init);
        Err(diags.report_internal_error(id.span(), "trying to create register in compile-time context"))
    }

    fn report_assignment(
        &mut self,
        ctx: &CompileItemContext,
        target: Spanned<Signal>,
        vars: &mut VariableValues,
    ) -> Result<(), ErrorGuaranteed> {
        let _ = vars;
        let diags = ctx.refs.diags;

        Err(diags.report_internal_error(
            target.span,
            "assigning to signal in compile-time context should not be possible",
        ))
    }

    fn block_domain(&self) -> BlockDomain {
        BlockDomain::CompileTime
    }

    fn condition_domains(&self) -> &[Spanned<ValueDomain>] {
        &[]
    }

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        if domain.inner != ValueDomain::CompileTime {
            throw!(diags.report_internal_error(
                domain.span,
                "trying to push non-compile-time condition domain in compile-time context"
            ))
        }
        let _ = (diags, domain);
        f(self)
    }

    fn with_implications<T>(
        &mut self,
        diags: &Diagnostics,
        implications: Vec<Implication>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        if !implications.is_empty() {
            throw!(diags.report_internal_error(self.span, "trying to push implications in compile-time context"))
        }
        f(self)
    }

    fn for_each_implication(&self, _: ValueVersioned, _: impl FnMut(&Implication)) {
        // do nothing
    }

    fn is_ir_context(&self) -> bool {
        false
    }

    fn check_ir_context(&self, diags: &Diagnostics, span: Span, access: &str) -> Result<(), ErrorGuaranteed> {
        let diag = Diagnostic::new("trying to access hardware in compile-time context")
            .add_error(span, format!("accessing {access} here"))
            .add_info(
                self.span,
                format!("compile-time context because {} must be compile-time", self.reason),
            )
            .finish();
        Err(diags.report(diag))
    }
}

// TODO rename to hardware
// TODO move to module?
// TODO move report_assignment into BlockKind
//   probably as part of the "coverage" tracking for combinatorial loops (=don't read before write)
pub struct IrBuilderExpressionContext<'a> {
    report_assignment: &'a mut dyn FnMut(&CompileItemContext, Spanned<Signal>) -> Result<(), ErrorGuaranteed>,
    block_kind: BlockKind<'a>,
    ir_variables: IrVariables,
    condition_domains: Vec<Spanned<ValueDomain>>,
    implications: Vec<Implication>,
}

#[derive(Debug)]
pub enum BlockKind<'a> {
    // real blocks
    Combinatorial {
        span_keyword: Span,
    },
    Clocked {
        span_keyword: Span,
        domain: Spanned<SyncDomain<DomainSignal>>,
        ir_registers: &'a mut IrRegisters,
        extra_registers: ExtraRegisters<'a>,
    },
    // misc
    InstancePortConnection {
        span_connection: Span,
    },
    WireValue {
        span_value: Span,
    },
}

#[derive(Debug)]
pub enum ExtraRegisters<'a> {
    NoReset,
    WithReset(&'a mut Vec<ExtraRegisterInit>),
}

impl<'a> IrBuilderExpressionContext<'a> {
    pub fn new(
        block_kind: BlockKind<'a>,
        report_assignment: &'a mut dyn FnMut(&CompileItemContext, Spanned<Signal>) -> Result<(), ErrorGuaranteed>,
    ) -> Self {
        Self {
            block_kind,
            report_assignment,
            ir_variables: IrVariables::new(),
            condition_domains: vec![],
            implications: vec![],
        }
    }

    pub fn finish(self) -> IrVariables {
        assert!(self.condition_domains.is_empty());
        self.ir_variables
    }
}

impl ExpressionContext for IrBuilderExpressionContext<'_> {
    type Block = IrBlock;

    fn new_ir_block(&self) -> Self::Block {
        IrBlock { statements: vec![] }
    }

    fn push_ir_statement(
        &self,
        diags: &Diagnostics,
        block: &mut Self::Block,
        stmt: Spanned<IrStatement>,
    ) -> Result<(), ErrorGuaranteed> {
        let _ = diags;
        block.statements.push(stmt);
        Ok(())
    }

    fn push_ir_statement_block(&self, block_parent: &mut Self::Block, block_inner: Spanned<Self::Block>) {
        // skip pushing empty blocks to slightly clean up the resulting IR
        if block_inner.inner.statements.is_empty() {
            return;
        }

        block_parent.statements.push(block_inner.map_inner(IrStatement::Block));
    }

    fn unwrap_ir_block(&self, diags: &Diagnostics, span: Span, block: Self::Block) -> Result<IrBlock, ErrorGuaranteed> {
        let _ = (diags, span);
        Ok(block)
    }

    fn new_ir_variable(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        info: IrVariableInfo,
    ) -> Result<IrVariable, ErrorGuaranteed> {
        let _ = (diags, span);
        Ok(self.ir_variables.push(info))
    }

    fn new_ir_register(
        &mut self,
        ctx: &CompileItemContext,
        diags: &Diagnostics,
        id: MaybeIdentifier,
        ty: HardwareType,
        init: Spanned<MaybeUndefined<IrExpression>>,
    ) -> Result<(IrRegister, SyncDomain<DomainSignal>), ErrorGuaranteed> {
        let span = id.span();

        let error_not_clocked = |block_span: Span, block_kind: &str| {
            let diag = Diagnostic::new("creating new registers is only allowed in a clocked block")
                .add_error(span, "attempt to create a register here")
                .add_info(block_span, format!("currently in this {block_kind}"))
                .finish();
            diags.report(diag)
        };

        match &mut self.block_kind {
            BlockKind::Clocked {
                span_keyword: _,
                domain,
                ir_registers,
                extra_registers,
            } => {
                let reg_info = IrRegisterInfo {
                    ty: ty.as_ir(),
                    debug_info_id: id.clone(),
                    debug_info_ty: ty.clone(),
                    debug_info_domain: domain.inner.to_diagnostic_string(ctx),
                };
                let reg = ir_registers.push(reg_info);

                match extra_registers {
                    ExtraRegisters::NoReset => match init.inner {
                        MaybeUndefined::Undefined => {
                            // we can't reset, but luckily we don't need to
                        }
                        MaybeUndefined::Defined(_) => {
                            let diag = Diagnostic::new("registers without reset cannot have an initial value")
                                .add_error(init.span, "attempt to create a register with init here")
                                .add_info(domain.span, "the current clocked block is defined without reset here")
                                .footer(
                                    Level::Help,
                                    "either add an reset to the block or use `undef` as the the initial value",
                                )
                                .finish();
                            let _ = diags.report(diag);
                        }
                    },
                    ExtraRegisters::WithReset(extra_registers) => {
                        match init.inner {
                            MaybeUndefined::Undefined => {
                                // we can reset but we don't need to
                            }
                            MaybeUndefined::Defined(init) => {
                                extra_registers.push(ExtraRegisterInit { span, reg, init });
                            }
                        }
                    }
                }

                Ok((reg, domain.inner))
            }
            &mut BlockKind::Combinatorial { span_keyword } => {
                Err(error_not_clocked(span_keyword, "combinatorial block"))
            }
            &mut BlockKind::InstancePortConnection { span_connection } => {
                Err(error_not_clocked(span_connection, "instance port connection"))
            }
            &mut BlockKind::WireValue { span_value } => Err(error_not_clocked(span_value, "wire value")),
        }
    }

    fn report_assignment(
        &mut self,
        ctx: &CompileItemContext,
        target: Spanned<Signal>,
        vars: &mut VariableValues,
    ) -> Result<(), ErrorGuaranteed> {
        // TODO remove this callback indirection,
        //   there's only one user that cares about it (module) and they just want a recording
        let err1 = (self.report_assignment)(ctx, target);
        let err2 = vars.signal_report_write(ctx.refs.diags, target);

        err1?;
        err2?;
        Ok(())
    }

    fn block_domain(&self) -> BlockDomain {
        match self.block_kind {
            BlockKind::Clocked { domain, .. } => BlockDomain::Clocked(domain),
            BlockKind::Combinatorial { .. }
            | BlockKind::InstancePortConnection { .. }
            | BlockKind::WireValue { .. } => BlockDomain::Combinatorial,
        }
    }

    fn condition_domains(&self) -> &[Spanned<ValueDomain>] {
        &self.condition_domains
    }

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        let _ = diags;

        let len_domains_before = self.condition_domains.len();
        self.condition_domains.push(domain);

        let result = f(self);

        assert_eq!(self.condition_domains.len(), len_domains_before + 1);
        self.condition_domains.pop();

        result
    }

    fn with_implications<T>(
        &mut self,
        diags: &Diagnostics,
        implications: Vec<Implication>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        let _ = diags;

        let len_implications_before = self.implications.len();
        let len_implications_added = implications.len();
        self.implications.extend(implications);

        let result = f(self);

        assert_eq!(
            self.implications.len(),
            len_implications_before + len_implications_added
        );
        self.implications.truncate(len_implications_before);

        result
    }

    fn for_each_implication(&self, value: ValueVersioned, mut f: impl FnMut(&Implication)) {
        for implication in &self.implications {
            if implication.value == value {
                f(implication);
            }
        }
    }

    fn is_ir_context(&self) -> bool {
        true
    }

    fn check_ir_context(&self, _: &Diagnostics, _: Span, _: &str) -> Result<(), ErrorGuaranteed> {
        Ok(())
    }
}
