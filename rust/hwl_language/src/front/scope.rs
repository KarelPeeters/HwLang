use annotate_snippets::Level;
use std::fmt::{Debug, Display, Formatter};

use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, DiagnosticResult, Diagnostics};
use crate::data::source::SourceDatabase;
use crate::new_index_type;
use crate::syntax::ast;
use crate::syntax::pos::Span;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use indexmap::map::IndexMap;

new_index_type!(pub Scope);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug)]
pub struct ScopeInfo<V> {
    span: Span,
    #[allow(dead_code)]
    scope: Scope,
    parent: Option<(Scope, Visibility)>,
    values: IndexMap<String, (V, Span, Visibility)>,
}

#[derive(Debug)]
pub struct ScopeFound<V> {
    pub defining_span: Span,
    pub value: V,
}

pub struct Scopes<V> {
    arena: Arena<Scope, ScopeInfo<V>>,
}

// TODO rethink: have a separate scope builder that needs all declarations to be provided before any lookups can start?
//   more safe and elegant, but makes "a bunch of nested scopes" like in generic params much slower
impl<V> Scopes<V> {
    pub fn default() -> Self {
        Self { arena: Arena::default() }
    }

    pub fn new_root(&mut self, span: Span) -> Scope {
        self.arena.push_with_index(|scope| ScopeInfo {
            span,
            scope,
            parent: None,
            values: Default::default(),
        })
    }

    pub fn new_child(&mut self, parent: Scope, span: Span, vis: Visibility) -> Scope {
        self.arena.push_with_index(|child| ScopeInfo {
            span,
            scope: child,
            parent: Some((parent, vis)),
            values: Default::default(),
        })
    }
}

impl<V> ScopeInfo<V> {
    // TODO make _local_ shadowing configurable: allowed, non-local allowed, not allowed
    // TODO make "identifier" string configurable
    pub fn declare<'a>(&mut self, diagnostics: &Diagnostics, id: &ast::Identifier, var: V, vis: Visibility) -> DiagnosticResult<()> {
        if let Some(&(_, prev_span, _)) = self.values.get(&id.string) {
            // TODO allow shadowing? for items and parameters no, but maybe for local variables yes?
            let err = Diagnostic::new("identifier declared twice")
                .add_info(prev_span, "previously declared here")
                .add_error(id.span, "declared again here")
                .finish();
            Err(diagnostics.report(err))
        } else {
            // only insert if we know the id is not declared yet, to avoid state mutation in case of an error
            self.values.insert_first(id.string.to_owned(), (var, id.span, vis));
            Ok(())
        }
    }

    pub fn maybe_declare(&mut self, diagnostics: &Diagnostics, id: &ast::MaybeIdentifier, var: V, vis: Visibility) -> DiagnosticResult<()> {
        match id {
            ast::MaybeIdentifier::Identifier(id) =>
                self.declare(diagnostics, id, var, vis),
            ast::MaybeIdentifier::Dummy(_) =>
                Ok(())
        }
    }

    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    ///
    /// If the item if found but not accessible, an error is logged but the value is still returned.
    /// This still allows typechecking to happen.
    pub fn find<'s>(
        &'s self,
        scopes: &'s Scopes<V>,
        database: &SourceDatabase,
        diagnostics: &Diagnostics,
        id: &ast::Identifier,
        vis: Visibility,
    ) -> DiagnosticResult<ScopeFound<&'s V>> {
        if let Some(&(ref value, value_span, value_vis)) = self.values.get(&id.string) {
            if !vis.can_access(value_vis) {
                let err = Diagnostic::new(format!("cannot access identifier `{}`", id.string))
                    .add_info(value_span, "identifier declared here")
                    .add_error(id.span, "not accessible here")
                    .footer(Level::Info, format!("Identifier was declared with visibility `{}`,\n but the access happens with visibility `{}`", value_vis, vis))
                    .finish();
                diagnostics.report_and_continue(err);
            }

            Ok(ScopeFound { defining_span: value_span, value })
        } else if let Some((parent, parent_vis)) = self.parent {
            // TODO does min access make sense?
            scopes[parent].find(scopes, database, diagnostics, id, Visibility::minimum_access(vis, parent_vis))
        } else {
            // TODO add fuzzy-matched suggestions as info
            let err = Diagnostic::new(format!("undeclared identifier `{}`", id.string))
                .add_error(id.span, "identifier not declared")
                .add_info(Span::empty_at(self.span.start), "searched in the scope starting here and its parents")
                .finish();
            Err(diagnostics.report(err))
        }
    }

    pub fn find_immediate_str(
        &self,
        diagnostics: &Diagnostics,
        id: &str,
        vis: Visibility,
    ) -> DiagnosticResult<ScopeFound<&V>> {
        if let Some(&(ref value, value_span, value_vis)) = self.values.get(id) {
            if vis.can_access(value_vis) {
                Ok(ScopeFound { defining_span: value_span, value })
            } else {
                let err = Diagnostic::new(format!("cannot access identifier `{}` externally", id))
                    .add_info(value_span, "identifier declared here")
                    .footer(Level::Info, format!("Identifier was declared with visibility `{}`,\n but the access happens with visibility `{}`", value_vis, vis))
                    .finish();
                Err(diagnostics.report(err))
            }
        } else {
            let err = Diagnostic::new(format!("undeclared identifier `{}`", id))
                .add_info(Span::empty_at(self.span.start), "searched in the scope starting here")
                .finish();
            Err(diagnostics.report(err))
        }
    }

    /// The amount of values declared in this scope without taking the parent scope into account.
    pub fn size(&self) -> usize {
        self.values.len()
    }
}

impl<V> std::ops::Index<Scope> for Scopes<V> {
    type Output = ScopeInfo<V>;
    fn index(&self, index: Scope) -> &Self::Output {
        &self.arena[index]
    }
}

impl<V> std::ops::IndexMut<Scope> for Scopes<V> {
    fn index_mut(&mut self, index: Scope) -> &mut Self::Output {
        &mut self.arena[index]
    }
}

impl Visibility {
    fn can_access(self, other: Visibility) -> bool {
        match (self, other) {
            (Visibility::Private, _) => true,
            (Visibility::Public, Visibility::Public) => true,
            (Visibility::Public, Visibility::Private) => false,
        }
    }

    fn minimum_access(self, other: Visibility) -> Visibility {
        match (self, other) {
            (Visibility::Private, Visibility::Private) => Visibility::Private,
            (Visibility::Public, _) | (_, Visibility::Public) => Visibility::Public,
        }
    }
}

impl Display for Visibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Public => write!(f, "public"),
            Visibility::Private => write!(f, "private"),
        }
    }
}