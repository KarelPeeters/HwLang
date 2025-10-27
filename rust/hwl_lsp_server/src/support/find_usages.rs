use crate::support::PosNotOnIdentifier;
use hwl_language::syntax::ast::{FileContent, GeneralIdentifier};
use hwl_language::syntax::pos::{HasSpan, Pos, Span};
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{DeclScope, EvaluatedId, SyntaxVisitor, syntax_visit};
use hwl_language::util::Never;
use itertools::Itertools;
use std::ops::ControlFlow;

// TODO make this cross-file
pub fn find_usages(source: &SourceDatabase, ast: &FileContent, pos: Pos) -> Result<Vec<Span>, PosNotOnIdentifier> {
    // find the full span for the current id if any
    let id_span = syntax_visit(source, ast, &mut FindDeclaredIdVisitor { pos });
    let id_span = match id_span {
        ControlFlow::Continue(()) => return Err(PosNotOnIdentifier),
        ControlFlow::Break(id_span) => id_span,
    };

    // find usages of the given id
    let mut usages = vec![];
    let mut visitor = FindUsagesVisitor {
        target_id_span: id_span,
        source,
        result: &mut usages,
    };
    let _ = syntax_visit(source, ast, &mut visitor);

    Ok(usages.iter().map(|id| id.span()).collect_vec())
}

struct FindDeclaredIdVisitor {
    pos: Pos,
}

impl SyntaxVisitor for FindDeclaredIdVisitor {
    type Break = Span;
    const SCOPE_DECLARE: bool = false;

    fn should_visit_span(&self, span: Span) -> bool {
        span.touches_pos(self.pos)
    }

    fn report_id_declare(&mut self, id: GeneralIdentifier) -> ControlFlow<Self::Break, ()> {
        if id.span().touches_pos(self.pos) {
            ControlFlow::Break(id.span())
        } else {
            ControlFlow::Continue(())
        }
    }
}

struct FindUsagesVisitor<'a> {
    target_id_span: Span,
    source: &'a SourceDatabase,
    result: &'a mut Vec<GeneralIdentifier>,
}

impl SyntaxVisitor for FindUsagesVisitor<'_> {
    type Break = Never;
    const SCOPE_DECLARE: bool = true;

    fn should_visit_span(&self, _: Span) -> bool {
        // TODO only visit spans that can possibly access the declared ID
        true
    }

    fn report_id_use(
        &mut self,
        scope: &DeclScope<'_>,
        id: GeneralIdentifier,
        id_eval: impl Fn(&SourceDatabase, GeneralIdentifier) -> EvaluatedId<&str>,
    ) -> ControlFlow<Self::Break, ()> {
        let found = scope.find(&id_eval(self.source, id));
        for span in found {
            if span == self.target_id_span {
                self.result.push(id);
            }
        }
        ControlFlow::Continue(())
    }
}
