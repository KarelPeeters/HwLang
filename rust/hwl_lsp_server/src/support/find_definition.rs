use hwl_language::syntax::ast::{FileContent, GeneralIdentifier};
use hwl_language::syntax::pos::{Pos, Span};
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{DeclScope, EvaluatedId, SyntaxVisitor, syntax_visit};
use std::ops::ControlFlow;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PosNotOnIdentifier;

pub fn find_definition(source: &SourceDatabase, ast: &FileContent, pos: Pos) -> Result<Vec<Span>, PosNotOnIdentifier> {
    let mut visitor = FindDeclarationVisitor { source, pos };
    let res = syntax_visit(source, ast, &mut visitor);
    match res {
        ControlFlow::Continue(()) => Err(PosNotOnIdentifier),
        ControlFlow::Break(found) => Ok(found),
    }
}

struct FindDeclarationVisitor<'a> {
    source: &'a SourceDatabase,
    pos: Pos,
}

impl SyntaxVisitor for FindDeclarationVisitor<'_> {
    type Break = Vec<Span>;
    const SCOPE_DECLARE: bool = true;

    fn should_visit_span(&self, span: Span) -> bool {
        span.touches_pos(self.pos)
    }

    fn report_id_use(
        &mut self,
        scope: &DeclScope<'_>,
        id: GeneralIdentifier,
        id_eval: impl Fn(&SourceDatabase, GeneralIdentifier) -> EvaluatedId<&str>,
    ) -> ControlFlow<Self::Break, ()> {
        ControlFlow::Break(scope.find(&id_eval(self.source, id)))
    }
}
