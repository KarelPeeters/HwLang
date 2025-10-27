use crate::support::PosNotOnIdentifier;
use crate::support::find_definition::find_definition;
use hwl_language::syntax::ast::{FileContent, GeneralIdentifier};
use hwl_language::syntax::pos::{HasSpan, Pos, Span};
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{DeclScope, EvaluatedId, SyntaxVisitor, syntax_visit};
use hwl_language::util::Never;
use indexmap::IndexSet;
use itertools::Itertools;
use std::ops::ControlFlow;

// TODO make this cross-file
pub fn find_usages(source: &SourceDatabase, ast: &FileContent, pos: Pos) -> Result<Vec<Span>, PosNotOnIdentifier> {
    // decide which id(s) to find the usages of
    //   prefer found definitions, they're the result of the position being on an identifier usage,
    //   which can be inside general identifier as an expression
    let target_ids = find_definition(source, ast, pos)
        .ok()
        .or_else(|| {
            syntax_visit(source, ast, &mut FindDeclaredIdVisitor { pos })
                .break_value()
                .map(|v| vec![v])
        })
        .ok_or(PosNotOnIdentifier)?;

    // find usages of the given id
    let mut usages = vec![];
    let mut visitor = FindUsagesVisitor {
        target_ids: target_ids.into_iter().collect(),
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
    target_ids: IndexSet<Span>,
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
            if self.target_ids.contains(&span) {
                self.result.push(id);
            }
        }
        ControlFlow::Continue(())
    }
}
