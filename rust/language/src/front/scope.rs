use std::fmt::Debug;

use indexmap::map::IndexMap;

use crate::syntax::ast;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug)]
pub struct Scope<'p, V> {
    parent: Option<(&'p Scope<'p, V>, Visibility)>,
    values: IndexMap<String, (V, Visibility)>,
}

pub type ScopeResult<T> = Result<T, ScopeError>;

#[must_use]
#[derive(Debug)]
pub enum ScopeError {
    IdentifierDeclaredTwice(ast::Identifier),
    UndeclaredIdentifier(ast::Identifier),
    CannotAccess(ast::Identifier),
}

impl<V: Debug> Scope<'_, V> {
    pub fn nest(&self, vis: Visibility) -> Scope<V> {
        Scope { parent: Some((self, vis)), values: Default::default() }
    }

    pub fn declare<'a>(&mut self, id: &ast::Identifier, var: V, vis: Visibility) -> ScopeResult<()> {
        if self.values.insert(id.string.to_owned(), (var, vis)).is_some() {
            // TODO allow shadowing? for items and parameters no, but maybe for local variables yes?
            Err(ScopeError::IdentifierDeclaredTwice(id.clone()))
        } else {
            Ok(())
        }
    }

    pub fn maybe_declare(&mut self, id: &ast::MaybeIdentifier, var: V, vis: Visibility) -> ScopeResult<()> {
        match id {
            ast::MaybeIdentifier::Identifier(id) =>
                self.declare(id, var, vis),
            ast::MaybeIdentifier::Dummy(_) =>
                Ok(())
        }
    }

    /// Declare a value with the given id. Panics if the id already exists in this scope.
    pub fn declare_str(&mut self, id: &str, var: V, vis: Visibility) {
        let prev = self.values.insert(id.to_owned(), (var, vis));

        if let Some(prev) = prev {
            panic!("Id '{}' already exists in this scope with value {:?}", id, prev)
        }
    }

    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    pub fn find<'a, 's>(&'s self, root: Option<&'s Self>, id: &'a ast::Identifier, vis: Visibility) -> ScopeResult<&V> {
        if let Some(&(ref s, s_vis)) = self.values.get(&id.string) {
            if vis.can_access(s_vis) {
                Ok(s)
            } else {
                Err(ScopeError::CannotAccess(id.clone()))
            }
        } else if let Some((p, p_vis)) = self.parent {
            // TODO does min access make sense?
            p.find(root, id, Visibility::minimum_access(vis, p_vis))
        } else if let Some(root) = root {
            // TODO do we need vis support here too?
            root.find(None, id, Visibility::Public)
        } else {
            Err(ScopeError::UndeclaredIdentifier(id.clone()))
        }
    }

    /// Find the given identifier in this scope without looking at the parent scope.
    pub fn find_immediate_str(&self, id: &str) -> Option<&V> {
        self.values.get(id).map(|(s, _)| s)
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