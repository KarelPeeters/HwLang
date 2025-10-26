use crate::syntax::ast::{FileContent, GeneralIdentifier};
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::SourceDatabase;
use crate::syntax::visitor::{DeclScope, EvaluatedId, SyntaxVisitor, syntax_visit};
use std::ops::ControlFlow;

// TODO document the bigger-picture approach taken, visit the entire AST with early existing,
//   build scopes by declaring everything on the way,
//   and when we finally encounter the identifier look it up in the current scopes

// TODO visitor (shudders) design sketch:
//   the following LSP features all need some kind of AST visiting:
//   * go to definition: walk down, declaring things in scopes, skip things outside of certain span, return single potential result
//   * find usages: walk down, declaring things in scopes, don't skip anything, collect multiple results
//   * expand selection: walk down, building nested tree of spans, skip things outside of certain span, return span stack (effectively a single result)
//   * folding ranges: walk down, building
//   * find the list of declared top-level IDs in a file? we don't really need the full visitor for that
//   * autocomplete: find the full set of identifiers that is in scope at a certain position

// TODO rename (and its variants)
#[derive(Debug, Eq, PartialEq)]
pub enum FindDefinition<S = Vec<Span>> {
    Found(S),
    PosNotOnIdentifier,
}

// TODO implement the opposite direction, "find usages"
// TODO follow imports instead of jumping to them
// TODO we can do better: in `if(_) { a } else { a }` a is not really conditional any more
// TODO generalize this "visitor", we also want to collect all usages, find the next selection span, find folding ranges, ...
// TODO maybe this should be moved to the LSP, the compiler itself really doesn't need this
// TODO use the real Scope for the file root, to reduce duplication and get a guaranteed match
pub fn find_definition(source: &SourceDatabase, ast: &FileContent, pos: Pos) -> FindDefinition {
    let mut visitor = FindDeclarationVisitor { source, pos };
    let res = syntax_visit(source, ast, &mut visitor);
    match res {
        ControlFlow::Continue(()) => FindDefinition::PosNotOnIdentifier,
        ControlFlow::Break(found) => FindDefinition::Found(found),
    }
}

struct FindDeclarationVisitor<'a> {
    source: &'a SourceDatabase,
    pos: Pos,
}

// TODO mode function bodies here instead of in the parent
impl SyntaxVisitor for FindDeclarationVisitor<'_> {
    type Break = Vec<Span>;
    const DECLARE: bool = true;

    fn filter_by_pos(&self) -> Option<Pos> {
        Some(self.pos)
    }

    fn report_id_usage(
        &mut self,
        scope: &DeclScope<'_>,
        id: GeneralIdentifier,
        id_eval: impl Fn(&SourceDatabase, GeneralIdentifier) -> EvaluatedId<&str>,
    ) -> ControlFlow<Self::Break, ()> {
        ControlFlow::Break(scope.find(&id_eval(self.source, id)))
    }
}
