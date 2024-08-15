use std::fmt::Debug;

use indexmap::map::IndexMap;

use crate::error::DiagnosticError;
use crate::front::driver::{DiagnosticAddable, SourceDatabase};
use crate::syntax::ast;
use crate::syntax::pos::Span;
use crate::throw;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug)]
pub struct Scope<'p, V> {
    parent: Option<(&'p Scope<'p, V>, Visibility)>,
    values: IndexMap<String, (V, Span, Visibility)>,
}

pub type ScopeResult<T> = Result<T, DiagnosticError>;

impl<V: Debug> Scope<'_, V> {
    pub fn nest(&self, vis: Visibility) -> Scope<V> {
        Scope { parent: Some((self, vis)), values: Default::default() }
    }

    pub fn declare<'a>(&mut self, database: &SourceDatabase, id: &ast::Identifier, var: V, vis: Visibility) -> ScopeResult<()> {
        if let Some(&(_, prev_span, _)) = self.values.get(&id.string) {
            // TODO allow shadowing? for items and parameters no, but maybe for local variables yes?

            let err = database.diagnostic("identifier declared twice")
                .snippet(prev_span)
                .add_info(id.span, "previously declared here")
                .finish()
                .snippet(id.span)
                .add_error(prev_span, "declared again here")
                .finish()
                .finish();
            throw!(err)
        } else {
            // only insert if we know the id is not declared yet, to avoid state mutation in case of an error
            let prev = self.values.insert(id.string.to_owned(), (var, id.span, vis));
            assert!(prev.is_none());
            Ok(())
        }
    }

    pub fn maybe_declare(&mut self, database: &SourceDatabase, id: &ast::MaybeIdentifier, var: V, vis: Visibility) -> ScopeResult<()> {
        match id {
            ast::MaybeIdentifier::Identifier(id) =>
                self.declare(database, id, var, vis),
            ast::MaybeIdentifier::Dummy(_) =>
                Ok(())
        }
    }

    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    pub fn find<'a, 's>(&'s self, database: &SourceDatabase, root: Option<&'s Self>, id: &'a ast::Identifier, vis: Visibility) -> ScopeResult<&V> {
        if let Some(&(ref value, value_span, value_vis)) = self.values.get(&id.string) {
            if vis.can_access(value_vis) {
                Ok(value)
            } else {
                let err = database.diagnostic("cannot access definition")
                    .snippet(value_span)
                    .add_info(value_span, "identifier declared here")
                    .finish()
                    .snippet(id.span)
                    .add_error(id.span, "not accessible here")
                    .finish()
                    .finish();
                throw!(err)
            }
        } else if let Some((parent, parent_vis)) = self.parent {
            // TODO does min access make sense?
            parent.find(database, root, id, Visibility::minimum_access(vis, parent_vis))
        } else if let Some(root) = root {
            // TODO do we need vis support here too?
            root.find(database, None, id, Visibility::Public)
        } else {
            let err = database.diagnostic("undeclared identifier")
                .snippet(id.span)
                .add_error(id.span, "identifier not declared")
                .finish()
                .finish();
            throw!(err)
        }
    }

    /// The amount of values declared in this scope without taking the parent scope into account.
    pub fn size(&self) -> usize {
        self.values.len()
    }
}

impl<'p, V> Default for Scope<'p, V> {
    fn default() -> Self {
        Self {
            parent: Default::default(),
            values: Default::default(),
        }
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