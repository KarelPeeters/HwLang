use crate::front::block::ElaboratedForHeader;
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::flow::Flow;
use crate::front::scope::{Scope, ScopedEntry};
use crate::syntax::ast::{ExtraList, ExtraListBlock, ExtraListItem};
use crate::syntax::pos::{HasSpan, Spanned};

pub struct ExtraScope<'a, 'b, 'c, 'd> {
    root_scope: Option<&'a Scope<'b>>,
    scope: &'c mut Scope<'d>,
}

impl<'a, 'b, 'c, 'd> ExtraScope<'a, 'b, 'c, 'd> {
    pub fn as_scope(&mut self) -> &mut Scope<'d> {
        self.scope
    }

    pub fn new_child<'e, 'f>(&self, child_scope: &'e mut Scope<'f>) -> ExtraScope<'_, '_, 'e, 'f> {
        ExtraScope {
            root_scope: Some(self.root_scope.unwrap_or(self.scope)),
            scope: child_scope,
        }
    }

    pub fn declare_root(&mut self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, entry: DiagResult<ScopedEntry>) {
        self.scope.declare(diags, id, entry);
        if let Some(root_scope) = self.root_scope {
            root_scope.declare_non_mut(diags, id, entry);
        }
    }
}

impl CompileItemContext<'_, '_> {
    pub fn elaborate_extra_list<'a, F: Flow, T>(
        &mut self,
        scope_parent: &mut Scope,
        flow: &mut F,
        list: &'a ExtraList<T>,
        f: &mut impl FnMut(&mut Self, &mut ExtraScope, &mut F, &'a T) -> DiagResult,
    ) -> DiagResult {
        let ExtraList { span: _, items } = list;
        let mut scope_extra = ExtraScope {
            root_scope: None,
            scope: scope_parent,
        };
        self.elaborate_extra_list_items(&mut scope_extra, flow, items, f)
    }

    pub fn elaborate_extra_list_block<'a, F: Flow, T>(
        &mut self,
        scope_parent: &mut ExtraScope,
        flow: &mut F,
        block: &'a ExtraListBlock<T>,
        f: &mut impl FnMut(&mut Self, &mut ExtraScope, &mut F, &'a T) -> DiagResult,
    ) -> DiagResult {
        let &ExtraListBlock { span, ref items } = block;

        let mut scope_child = scope_parent.scope.new_child(span);
        let mut scope_child = scope_parent.new_child(&mut scope_child);
        self.elaborate_extra_list_items(&mut scope_child, flow, items, f)
    }

    fn elaborate_extra_list_items<'a, F: Flow, T: 'a>(
        &mut self,
        scope: &mut ExtraScope,
        flow: &mut F,
        items: &'a [ExtraListItem<T>],
        f: &mut impl FnMut(&mut Self, &mut ExtraScope, &mut F, &'a T) -> DiagResult,
    ) -> DiagResult {
        let refs = self.refs;

        for item in items {
            match item {
                ExtraListItem::Leaf(leaf) => f(self, scope, flow, leaf)?,
                ExtraListItem::Declaration(decl) => {
                    self.eval_and_declare_declaration(scope.as_scope(), flow, decl)?;
                }
                ExtraListItem::If(stmt) => {
                    let block = self.compile_if_statement_choose_block(scope.as_scope(), flow, stmt)?;
                    if let Some(block) = block {
                        self.elaborate_extra_list_block(scope, flow, block, f)?;
                    }
                }
                ExtraListItem::Match(stmt) => {
                    let (declare, block) = self.compile_match_statement_choose_branch(scope.as_scope(), flow, stmt)?;

                    let mut scope_child = scope.scope.new_child(block.span);
                    let mut scope_child = scope.new_child(&mut scope_child);

                    if let Some(declare) = declare {
                        declare.declare(refs, scope_child.as_scope(), flow)?;
                    }

                    self.elaborate_extra_list_block(&mut scope_child, flow, block, f)?;
                }
                ExtraListItem::For(stmt) => {
                    let ElaboratedForHeader { index_ty, iter } =
                        self.elaborate_for_statement_header(scope.as_scope(), flow, stmt)?;

                    for index_value in iter {
                        refs.check_should_stop(stmt.span_keyword)?;

                        let mut scope_iter = scope.scope.new_child(stmt.span());
                        let mut scope_iter = scope.new_child(&mut scope_iter);

                        self.elaborate_for_statement_iteration(
                            scope_iter.as_scope(),
                            flow,
                            stmt,
                            &index_ty,
                            index_value,
                        )?;

                        self.elaborate_extra_list_block(&mut scope_iter, flow, &stmt.body, f)?;
                    }
                }
            }
        }
        Ok(())
    }
}
