use annotate_snippets::Level;
use std::fmt::{Debug, Display, Formatter};

use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
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
    values: IndexMap<String, DeclaredValue<V>>,
}

#[derive(Debug)]
pub enum DeclaredValue<V> {
    Once { value: V, span: Span, vis: Visibility },
    Multiple { spans: Vec<Span>, err: ErrorGuaranteed },
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
    /// Declare a value in this scope.
    ///
    /// Allows shadowing identifiers in the parent scope, but not in the local scope.
    ///
    /// This function always appears to succeed, errors are instead reported as diagnostics.
    /// This also tracks identifiers that have erroneously been declared multiple times,
    /// so that [Scope::find] can return an error for those cases.
    pub fn declare<'a>(&mut self, diagnostics: &Diagnostics, id: &ast::Identifier, value: V, vis: Visibility) {
        if let Some(declared) = self.values.get_mut(&id.string) {
            // get all spans
            let mut spans = match declared {
                DeclaredValue::Once { value: _, span, vis: _ } => vec![*span],
                DeclaredValue::Multiple { spans, err: _ } => std::mem::take(spans),
            };

            // report error
            let mut diag = Diagnostic::new("identifier declared multiple times");
            for span in &spans {
                diag = diag.add_info(*span, "previously declared here");
            }
            let diag = diag.add_error(id.span, "declared again here").finish();
            let err = diagnostics.report(diag);

            // insert error value into scope to avoid downstream errors
            //   caused by only considering the first declared value
            spans.push(id.span);
            *declared = DeclaredValue::Multiple { spans, err }
        } else {
            let declared = DeclaredValue::Once { value, span: id.span, vis };
            self.values.insert_first(id.string.to_owned(), declared);
        }
    }

    pub fn maybe_declare(&mut self, diagnostics: &Diagnostics, id: &ast::MaybeIdentifier, var: V, vis: Visibility) {
        match id {
            ast::MaybeIdentifier::Identifier(id) =>
                self.declare(diagnostics, id, var, vis),
            ast::MaybeIdentifier::Dummy(_) => {}
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
        diagnostics: &Diagnostics,
        id: &ast::Identifier,
        vis: Visibility,
    ) -> Result<ScopeFound<&'s V>, ErrorGuaranteed> {
        if let Some(declared) = self.values.get(&id.string) {
            // check declared exactly once
            let (value, value_span, value_vis) = match *declared {
                DeclaredValue::Once { ref value, span, vis } => (value, span, vis),
                DeclaredValue::Multiple { spans: _, err } => return Err(err),
            };

            // check access
            if !vis.can_access(value_vis) {
                let err = Diagnostic::new(format!("cannot access identifier `{}`", id.string))
                    .add_info(value_span, "identifier declared here")
                    .add_error(id.span, "not accessible here")
                    .footer(Level::Info, format!("Identifier was declared with visibility `{}`,\n but the access happens with visibility `{}`", value_vis, vis))
                    .finish();
                diagnostics.report(err);
            }

            Ok(ScopeFound { defining_span: value_span, value })
        } else if let Some((parent, parent_vis)) = self.parent {
            // TODO does min access make sense?
            scopes[parent].find(scopes, diagnostics, id, Visibility::minimum_access(vis, parent_vis))
        } else {
            // TODO add fuzzy-matched suggestions as info
            let err = Diagnostic::new(format!("undeclared identifier `{}`", id.string))
                .add_error(id.span, "identifier not declared")
                .add_info(Span::empty_at(self.span.start), "searched in the scope starting here, and its parents")
                .finish();
            Err(diagnostics.report(err))
        }
    }

    // TODO share common code with find, the only difference is a missing span and parent lookup
    pub fn find_immediate_str(
        &self,
        diagnostics: &Diagnostics,
        id: &str,
        vis: Visibility,
    ) -> Result<ScopeFound<&V>, ErrorGuaranteed> {
        if let Some(declared) = self.values.get(id) {
            // check declared exactly once
            let (value, value_span, value_vis) = match *declared {
                DeclaredValue::Once { ref value, span, vis } => (value, span, vis),
                DeclaredValue::Multiple { spans: _, err } => return Err(err),
            };

            // check vis
            if !vis.can_access(value_vis) {
                let err = Diagnostic::new(format!("cannot access identifier `{}` externally", id))
                    .add_info(value_span, "identifier declared here")
                    .footer(Level::Info, format!("Identifier was declared with visibility `{}`,\n but the access happens with visibility `{}`", value_vis, vis))
                    .finish();
                return Err(diagnostics.report(err));
            }

            Ok(ScopeFound { defining_span: value_span, value })
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
