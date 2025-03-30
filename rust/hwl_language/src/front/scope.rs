use annotate_snippets::Level;
use std::fmt::{Debug, Display, Formatter};

use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::misc::ScopedEntry;
use crate::new_index_type;
use crate::syntax::ast;
use crate::syntax::ast::Identifier;
use crate::syntax::pos::Span;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use indexmap::map::{Entry, IndexMap};

#[derive(Debug, Copy, Clone)]
pub enum Scope {
    File(ScopeFile),
    Inner(ScopeInner),
}

new_index_type!(pub ScopeFile);
new_index_type!(pub ScopeInner);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug)]
pub struct ScopeInfo {
    span: Span,
    parent: Option<(Scope, Visibility)>,
    values: IndexMap<String, DeclaredValue>,
}

#[derive(Debug)]
pub enum DeclaredValue {
    Once {
        value: Result<ScopedEntry, ErrorGuaranteed>,
        span: Span,
        vis: Visibility,
    },
    Multiple {
        spans: Vec<Span>,
        err: ErrorGuaranteed,
    },
}

#[derive(Debug)]
pub struct ScopeFound<'s> {
    pub defining_span: Span,
    pub value: &'s ScopedEntry,
}

pub struct Scopes<'a> {
    file_scopes: &'a Arena<ScopeFile, ScopeInfo>,
    inner_scopes: Arena<ScopeInner, ScopeInfo>,
}

// TODO rethink: have a separate scope builder that needs all declarations to be provided before any lookups can start?
//   more safe and elegant, but makes "a bunch of nested scopes" like in generic params much slower
impl<'a> Scopes<'a> {
    pub fn new(file_scopes: &'a Arena<ScopeFile, ScopeInfo>) -> Self {
        Self {
            file_scopes,
            inner_scopes: Arena::default(),
        }
    }

    pub fn new_child(&mut self, parent: impl Into<Scope>, span: Span, vis: Visibility) -> ScopeInner {
        let parent = parent.into();
        self.inner_scopes.push(ScopeInfo {
            span,
            parent: Some((parent, vis)),
            values: Default::default(),
        })
    }
}

impl ScopeInfo {
    pub fn new(span: Span, parent: Option<(Scope, Visibility)>) -> Self {
        Self {
            span,
            parent,
            values: Default::default(),
        }
    }

    /// Declare a value in this scope.
    ///
    /// Allows shadowing identifiers in the parent scope, but not in the local scope.
    ///
    /// This function always appears to succeed, errors are instead reported as diagnostics.
    /// This also tracks identifiers that have erroneously been declared multiple times,
    /// so that [Scope::find] can return an error for those cases.
    pub fn declare(
        &mut self,
        diagnostics: &Diagnostics,
        id: &Identifier,
        value: Result<ScopedEntry, ErrorGuaranteed>,
        vis: Visibility,
    ) {
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
            let declared = DeclaredValue::Once {
                value,
                span: id.span,
                vis,
            };
            self.values.insert_first(id.string.to_owned(), declared);
        }
    }

    pub fn declare_already_checked(
        &mut self,
        diagnostics: &Diagnostics,
        id: String,
        span: Span,
        value: Result<ScopedEntry, ErrorGuaranteed>,
    ) -> Result<(), ErrorGuaranteed> {
        match self.values.entry(id.clone()) {
            Entry::Occupied(_) => {
                Err(diagnostics.report_internal_error(span, format!("identifier `{}` already declared", id)))
            }
            Entry::Vacant(entry) => {
                entry.insert(DeclaredValue::Once {
                    value,
                    span,
                    vis: Visibility::Private,
                });
                Ok(())
            }
        }
    }

    pub fn maybe_declare(
        &mut self,
        diagnostics: &Diagnostics,
        id: ast::MaybeIdentifier<&Identifier>,
        var: Result<ScopedEntry, ErrorGuaranteed>,
        vis: Visibility,
    ) {
        match id {
            ast::MaybeIdentifier::Identifier(id) => self.declare(diagnostics, id, var, vis),
            ast::MaybeIdentifier::Dummy(_) => {}
        }
    }

    // TODO make this immediately create the error entry value if not found
    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    ///
    /// If the item if found but not accessible, an error is logged but the value is still returned.
    /// This still allows typechecking to happen.
    pub fn find<'s>(
        &'s self,
        scopes: &'s Scopes,
        diagnostics: &Diagnostics,
        id: &Identifier,
        vis: Visibility,
    ) -> Result<ScopeFound<'s>, ErrorGuaranteed> {
        self.find_impl(scopes, diagnostics, id, vis, self.span)
    }

    fn find_impl<'s>(
        &'s self,
        scopes: &'s Scopes,
        diagnostics: &Diagnostics,
        id: &Identifier,
        vis: Visibility,
        initial_scope_span: Span,
    ) -> Result<ScopeFound<'s>, ErrorGuaranteed> {
        if let Some(declared) = self.values.get(&id.string) {
            // check declared exactly once
            let (value, value_span, value_vis) = match *declared {
                DeclaredValue::Once { ref value, span, vis } => (value.as_ref_ok()?, span, vis),
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

            Ok(ScopeFound {
                defining_span: value_span,
                value,
            })
        } else if let Some((parent, parent_vis)) = self.parent {
            // TODO does min access make sense?
            scopes[parent].find_impl(
                scopes,
                diagnostics,
                id,
                Visibility::minimum_access(vis, parent_vis),
                initial_scope_span,
            )
        } else {
            // TODO add fuzzy-matched suggestions as info
            // TODO insert identifier into the current scope to suppress downstream errors
            let err = Diagnostic::new(format!("undeclared identifier `{}`", id.string))
                .add_error(id.span, "identifier not declared")
                .add_info(
                    Span::empty_at(initial_scope_span.start),
                    "searched in the scope starting here, and its parents",
                )
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
    ) -> Result<ScopeFound, ErrorGuaranteed> {
        if let Some(declared) = self.values.get(id) {
            // check declared exactly once
            let (value, value_span, value_vis) = match *declared {
                DeclaredValue::Once { ref value, span, vis } => (value.as_ref_ok()?, span, vis),
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

            Ok(ScopeFound {
                defining_span: value_span,
                value,
            })
        } else {
            // TODO insert identifier into the current scope to suppress downstream errors
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

impl std::ops::Index<Scope> for Scopes<'_> {
    type Output = ScopeInfo;
    fn index(&self, index: Scope) -> &Self::Output {
        match index {
            Scope::File(index) => &self.file_scopes[index],
            Scope::Inner(index) => &self.inner_scopes[index],
        }
    }
}

impl std::ops::Index<ScopeInner> for Scopes<'_> {
    type Output = ScopeInfo;
    fn index(&self, index: ScopeInner) -> &Self::Output {
        &self.inner_scopes[index]
    }
}

impl std::ops::IndexMut<ScopeInner> for Scopes<'_> {
    fn index_mut(&mut self, index: ScopeInner) -> &mut Self::Output {
        &mut self.inner_scopes[index]
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

impl From<ScopeInner> for Scope {
    fn from(inner: ScopeInner) -> Self {
        Scope::Inner(inner)
    }
}

impl From<ScopeFile> for Scope {
    fn from(file: ScopeFile) -> Self {
        Scope::File(file)
    }
}
