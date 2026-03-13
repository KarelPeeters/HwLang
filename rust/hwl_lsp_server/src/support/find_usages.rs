use crate::support::PosNotOnIdentifier;
use crate::support::find_definition::find_definition;
use hwl_language::syntax::ast::{FileContent, GeneralIdentifier, ParameterSelf};
use hwl_language::syntax::pos::{HasSpan, Pos, Span, Spanned};
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{SelfExpression, SyntaxVisitor, syntax_visit};
use hwl_language::util::{Never, ResultNeverExt};
use indexmap::IndexSet;
use itertools::Either;
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
    let mut usage_spans = vec![];
    let mut visitor = FindUsagesVisitor {
        target_ids: target_ids.into_iter().collect(),
        usage_spans: &mut usage_spans,
    };
    syntax_visit(source, ast, &mut visitor).remove_never();

    Ok(usage_spans)
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

    fn report_id_declare(&mut self, id: Either<GeneralIdentifier, ParameterSelf>) -> ControlFlow<Self::Break, ()> {
        let id_span = id.span();
        if id_span.touches_pos(self.pos) {
            ControlFlow::Break(id_span)
        } else {
            ControlFlow::Continue(())
        }
    }
}

struct FindUsagesVisitor<'a> {
    target_ids: IndexSet<Span>,
    usage_spans: &'a mut Vec<Span>,
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
        id: Either<GeneralIdentifier, Spanned<SelfExpression>>,
        scope_find_id: impl Fn() -> Vec<Span>,
    ) -> ControlFlow<Self::Break, ()> {
        for def_span in scope_find_id() {
            if self.target_ids.contains(&def_span) {
                self.usage_spans.push(id.span());
            }
        }
        ControlFlow::Continue(())
    }
}
