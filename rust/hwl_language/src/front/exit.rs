use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::types::Type;
use crate::syntax::pos::Span;

pub struct ExitStack<'r> {
    inside_block_expression: Option<InsideBlockExpression>,
    return_info: Option<&'r mut ReturnEntry<'r>>,
    stack: Vec<LoopEntry>,
}

#[derive(Debug, Copy, Clone)]
pub struct InsideBlockExpression(pub Span);

pub struct ReturnEntry<'r> {
    pub return_type: &'r Type,
    // TODO
    // return_flag: IrVariable,
    // return_any: bool,
    // return_value: Variable,
}

pub struct LoopEntry {
    // TODO
    // break_flag: IrVariable,
    // break_any: bool,
    //
    // continue_flag: IrVariable,
    // continue_any: bool,
}

impl<'r> ExitStack<'r> {
    pub fn new_root() -> Self {
        Self {
            inside_block_expression: None,
            return_info: None,
            stack: vec![],
        }
    }

    pub fn new_in_function(return_info: &'r mut ReturnEntry<'r>) -> Self {
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
}

impl<'r> ExitStack<'r> {
    pub fn with_loop_entry<R>(&mut self, f: impl FnOnce(&mut ExitStack) -> R) -> (LoopEntry, R) {
        self.stack.push(LoopEntry {});
        let result = f(self);
        let entry = self.stack.pop().unwrap();
        (entry, result)
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
            Some(entry) => Ok(*entry),
        }
    }

    pub fn innermost_loop_known(&mut self) -> &mut LoopEntry {
        self.stack.last_mut().unwrap()
    }

    pub fn innermost_loop(
        &mut self,
        diags: &Diagnostics,
        span_reason: Span,
        reason: &str,
    ) -> DiagResult<&mut LoopEntry> {
        match self.stack.last_mut() {
            None => match self.inside_block_expression {
                None => Err(diags.report_simple(
                    format!("{} can only be used inside a loop", reason),
                    span_reason,
                    format!("attempt to {} here", reason),
                )),
                Some(InsideBlockExpression(span_block)) => {
                    let d = Diagnostic::new_todo(format!("{} inside block expression", reason))
                        .add_error(span_reason, format!("attempt to {} here", reason))
                        .add_info(span_block, "inside this block expression")
                        .finish();
                    Err(diags.report(d))
                }
            },
            Some(entry) => Ok(entry),
        }
    }
}
