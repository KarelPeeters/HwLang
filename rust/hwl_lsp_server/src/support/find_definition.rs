use crate::support::PosNotOnIdentifier;
use hwl_language::syntax::ast::{FileContent, GeneralIdentifier};
use hwl_language::syntax::pos::{Pos, Span, Spanned};
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{SelfExpression, SyntaxVisitor, syntax_visit};
use itertools::Either;
use std::ops::ControlFlow;

pub fn find_definition(source: &SourceDatabase, ast: &FileContent, pos: Pos) -> Result<Vec<Span>, PosNotOnIdentifier> {
    let mut visitor = FindDeclarationVisitor { pos };
    let res = syntax_visit(source, ast, &mut visitor);
    match res {
        ControlFlow::Continue(()) => Err(PosNotOnIdentifier),
        ControlFlow::Break(found) => Ok(found),
    }
}

struct FindDeclarationVisitor {
    pos: Pos,
}

impl SyntaxVisitor for FindDeclarationVisitor {
    type Break = Vec<Span>;
    const SCOPE_DECLARE: bool = true;

    fn should_visit_span(&self, span: Span) -> bool {
        span.touches_pos(self.pos)
    }

    fn report_id_use(
        &mut self,
        _: Either<GeneralIdentifier, Spanned<SelfExpression>>,
        scope_find_id: impl Fn() -> Vec<Span>,
    ) -> ControlFlow<Self::Break, ()> {
        ControlFlow::Break(scope_find_id())
    }
}
