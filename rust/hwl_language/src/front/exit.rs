use crate::front::block::EarlyExitKind;
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::ValueDomain;
use crate::front::flow::{Flow, FlowHardware, FlowKind, Variable};
use crate::front::types::Type;
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrLargeArena, IrStatement, IrType,
    IrVariable, IrVariableInfo,
};
use crate::syntax::pos::{Span, Spanned};

pub struct ExitStack<'r> {
    inside_block_expression: Option<InsideBlockExpression>,
    return_info: Option<ReturnEntry<'r>>,
    stack: Vec<LoopEntry>,
}

#[derive(Debug, Copy, Clone)]
pub struct InsideBlockExpression(pub Span);

#[derive(Debug)]
pub struct ReturnEntry<'r> {
    pub span_function_decl: Span,

    pub return_type: Option<Spanned<&'r Type>>,
    pub return_var: Option<Variable>,

    pub kind: ReturnEntryKind,
}

#[derive(Debug)]
pub enum ReturnEntryKind {
    Compile,
    Hardware(ReturnEntryHardware),
}

#[derive(Debug)]
pub struct ReturnEntryHardware {
    pub return_flag: ExitFlag,
}

#[derive(Debug)]
pub enum LoopEntry {
    Compile,
    Hardware(LoopEntryHardware),
}

#[derive(Debug)]
pub struct LoopEntryHardware {
    pub break_flag: ExitFlag,
    pub continue_flag: ExitFlag,
}

#[derive(Debug)]
pub struct ExitFlag {
    var: IrVariable,
    domain: ValueDomain,

    counter_set: u64,
    counter_clear: u64,
}

impl LoopEntry {
    // TODO use Variable instead of IrVariable to get const prop, implications and joining for free
    pub fn new(flow: &mut impl Flow, span_keyword: Span) -> LoopEntry {
        match flow.kind_mut() {
            FlowKind::Compile(_) => LoopEntry::Compile,
            FlowKind::Hardware(flow) => {
                let entry = LoopEntryHardware {
                    break_flag: ExitFlag::new(flow, span_keyword, EarlyExitKind::Break),
                    continue_flag: ExitFlag::new(flow, span_keyword, EarlyExitKind::Continue),
                };

                LoopEntry::Hardware(entry)
            }
        }
    }
}

impl ExitFlag {
    pub fn new(flow: &mut FlowHardware, span: Span, kind: EarlyExitKind) -> ExitFlag {
        // crate variable
        let name = match kind {
            EarlyExitKind::Return => "flag_function_return",
            EarlyExitKind::Break => "flag_loop_break",
            EarlyExitKind::Continue => "flag_loop_continue",
        };
        let info = IrVariableInfo {
            ty: IrType::Bool,
            debug_info_span: span,
            debug_info_id: Some(name.to_owned()),
        };
        let var = flow.new_ir_variable(info);

        // initialize variable
        let stmt = IrStatement::Assign(IrAssignmentTarget::variable(var), IrExpression::Bool(false));
        flow.push_ir_statement(Spanned::new(span, stmt));

        // construct wrapper
        Self {
            var,
            domain: ValueDomain::Const,
            counter_set: 0,
            counter_clear: 0,
        }
    }

    pub fn clear(&mut self, flow: &mut FlowHardware, span: Span) {
        if self.counter_set > self.counter_clear {
            let stmt = IrStatement::Assign(IrAssignmentTarget::variable(self.var), IrExpression::Bool(false));
            flow.push_ir_statement(Spanned::new(span, stmt));

            self.counter_clear = self.counter_set;
        }
    }

    pub fn set(&mut self, cond_domain: ValueDomain, block: &mut IrBlock, span: Span) {
        let stmt = IrStatement::Assign(IrAssignmentTarget::variable(self.var), IrExpression::Bool(true));
        block.statements.push(Spanned::new(span, stmt));

        self.domain = self.domain.join(cond_domain);

        self.counter_set += 1;
    }

    pub fn get(&self) -> Option<(ValueDomain, IrVariable)> {
        if self.counter_set > self.counter_clear {
            Some((self.domain, self.var))
        } else {
            None
        }
    }
}

impl<'r> ExitStack<'r> {
    pub fn new_root() -> Self {
        Self {
            inside_block_expression: None,
            return_info: None,
            stack: vec![],
        }
    }

    pub fn new_in_function(return_info: ReturnEntry<'r>) -> Self {
        Self {
            inside_block_expression: None,
            return_info: Some(return_info),
            stack: vec![],
        }
    }

    pub fn new_in_block_expression(span: Span) -> Self {
        Self {
            inside_block_expression: Some(InsideBlockExpression(span)),
            return_info: None,
            stack: vec![],
        }
    }

    pub fn early_exit_condition(&mut self, large: &mut IrLargeArena) -> Option<(ValueDomain, IrExpression)> {
        // collect early-exit flags
        let mut exit_domain = ValueDomain::Const;
        let mut exit_flags = vec![];
        let mut add_flag = |flag: &ExitFlag| {
            if let Some((domain, flag)) = flag.get() {
                exit_domain = exit_domain.join(domain);
                exit_flags.push(flag);
            }
        };
        if let Some(entry) = self.return_info_option()
            && let ReturnEntryKind::Hardware(entry) = &entry.kind
        {
            add_flag(&entry.return_flag);
        }
        if let Some(LoopEntry::Hardware(entry)) = self.innermost_loop_option() {
            add_flag(&entry.break_flag);
            add_flag(&entry.continue_flag);
        }

        // if any flags are set, reduce to single boolean expression
        let exit_expression = exit_flags
            .into_iter()
            .map(IrExpression::Variable)
            .reduce(|a, b| large.push_expr(IrExpressionLarge::BoolBinary(IrBoolBinaryOp::Or, a, b)));
        exit_expression.map(|expr| (exit_domain, expr))
    }

    pub fn with_loop_entry<R>(&mut self, entry: LoopEntry, f: impl FnOnce(&mut ExitStack) -> R) -> R {
        self.stack.push(entry);
        let result = f(self);
        self.stack.pop().unwrap();
        result
    }

    pub fn return_info_option(&mut self) -> Option<&mut ReturnEntry<'r>> {
        self.return_info.as_mut()
    }

    pub fn return_info(&mut self, diags: &Diagnostics, span_return: Span) -> DiagResult<&mut ReturnEntry<'r>> {
        match &mut self.return_info {
            None => match self.inside_block_expression {
                None => Err(diags.report_simple(
                    "return can only be used inside a function",
                    span_return,
                    "attempt to return here",
                )),
                Some(InsideBlockExpression(span_block)) => {
                    let d = Diagnostic::new_todo("return inside block expression")
                        .add_error(span_return, "attempt to return here")
                        .add_info(span_block, "inside this block expression")
                        .finish();
                    Err(diags.report(d))
                }
            },
            Some(entry) => Ok(entry),
        }
    }

    pub fn innermost_loop_option(&mut self) -> Option<&mut LoopEntry> {
        self.stack.last_mut()
    }

    pub fn innermost_loop(
        &mut self,
        diags: &Diagnostics,
        span_reason: Span,
        reason: &str,
    ) -> DiagResult<&mut LoopEntry> {
        let inside_block_expression = self.inside_block_expression;

        self.innermost_loop_option()
            .ok_or_else(|| match inside_block_expression {
                None => diags.report_simple(
                    format!("{} can only be used inside a loop", reason),
                    span_reason,
                    format!("attempt to {} here", reason),
                ),
                Some(InsideBlockExpression(span_block)) => {
                    let d = Diagnostic::new_todo(format!("{} inside block expression", reason))
                        .add_error(span_reason, format!("attempt to {} here", reason))
                        .add_info(span_block, "inside this block expression")
                        .finish();
                    diags.report(d)
                }
            })
    }
}
