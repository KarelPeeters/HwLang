use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, Diagnostics};
use crate::front::flow::{FailedCaptureReason, Variable};
use crate::front::signal::{Port, PortInterface, Register, Wire, WireInterface};
use crate::syntax::ast::MaybeIdentifier;
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::source::FileId;
use crate::util::ResultExt;
use indexmap::map::{Entry, IndexMap};
use std::borrow::Borrow;
use std::fmt::Debug;

// TODO use string interning to avoid a bunch of string equality checks and hashing
#[derive(Debug)]
pub struct Scope<'p> {
    span: Span,
    parent: ScopeParent<'p>,
    content: ScopeContent,
}

#[derive(Debug, Copy, Clone)]
pub enum ScopedEntry {
    /// Indirection though an item, the item should be evaluated.
    Item(AstRefItem),
    /// A named value: port, register, wire, variable.
    /// These are not fully evaluated immediately, they might be used symbolically
    ///   as assignment targets or in domain expressions.
    Named(NamedValue),
}

#[derive(Debug, Copy, Clone)]
pub enum NamedValue {
    Variable(Variable),
    Port(Port),
    PortInterface(PortInterface),
    Wire(Wire),
    WireInterface(WireInterface),
    Register(Register),
}

#[derive(Debug, Copy, Clone)]
pub enum ScopeParent<'p> {
    /// Parent
    Some(&'p Scope<'p>),
    /// No parent, this is a file scope.
    None(FileId),
}

#[derive(Debug)]
pub struct ScopeContent {
    values: IndexMap<String, DeclaredValue>,
    any_id_err: DiagResult,
}

// TODO simplify all of this: we might only only need to report errors on the first re-declaration,
//   which means we can remove that branch entirely
#[derive(Debug)]
enum DeclaredValue {
    Once { value: DiagResult<ScopedEntry>, span: Span },
    Multiple { spans: Vec<Span>, err: DiagError },
    FailedCapture(Span, FailedCaptureReason),
    Error(DiagError),
}

#[derive(Debug, Copy, Clone)]
pub enum DeclaredValueSingle<S = ScopedEntry> {
    Value { span: Span, value: S },
    FailedCapture(Span, FailedCaptureReason),
    Error(DiagError),
}

#[derive(Debug)]
pub struct ScopeFound {
    pub defining_span: Span,
    pub value: ScopedEntry,
}

#[derive(Debug)]
pub enum TryScopeFound {
    Found(ScopeFound),
    NotFoundAnyIdErr(DiagResult),
}

impl<'p> Scope<'p> {
    pub fn new_root(span: Span, file: FileId) -> Self {
        Scope {
            span,
            parent: ScopeParent::None(file),
            content: ScopeContent {
                values: IndexMap::new(),
                any_id_err: Ok(()),
            },
        }
    }

    pub fn new_child(span: Span, parent: &'p Scope<'p>) -> Self {
        Scope {
            span,
            parent: ScopeParent::Some(parent),
            content: ScopeContent {
                values: IndexMap::new(),
                any_id_err: Ok(()),
            },
        }
    }

    pub fn restore_child_from_content(span: Span, parent: &'p Scope<'p>, values: ScopeContent) -> Self {
        Scope {
            span,
            parent: ScopeParent::Some(parent),
            content: values,
        }
    }

    pub fn parent(&self) -> ScopeParent<'p> {
        self.parent
    }

    pub fn into_content(self) -> ScopeContent {
        self.content
    }

    pub fn immediate_entries(&self) -> impl Iterator<Item = (&str, DeclaredValueSingle<ScopedEntry>)> {
        self.content.values.iter().map(|(k, v)| {
            let v = match *v {
                DeclaredValue::Once { value, span } => match value {
                    Ok(value) => DeclaredValueSingle::Value { span, value },
                    Err(e) => DeclaredValueSingle::Error(e),
                },
                DeclaredValue::Multiple { spans: _, err } => DeclaredValueSingle::Error(err),
                DeclaredValue::FailedCapture(span, reason) => DeclaredValueSingle::FailedCapture(span, reason),
                DeclaredValue::Error(err) => DeclaredValueSingle::Error(err),
            };
            (k.as_str(), v)
        })
    }

    pub fn has_immediate_entry(&self, id: &str) -> bool {
        self.content.values.contains_key(id)
    }

    /// Declare a value in this scope.
    ///
    /// Allows shadowing identifiers in the parent scope, but not in the local scope.
    ///
    /// This function always appears to succeed, errors are instead reported as diags.
    /// This also tracks identifiers that have erroneously been declared multiple times,
    /// so that [Scope::find] can return an error for those cases.
    pub fn declare(
        &mut self,
        diags: &Diagnostics,
        id: DiagResult<Spanned<impl Borrow<str>>>,
        value: DiagResult<ScopedEntry>,
    ) {
        let id = id.as_ref_ok().map(|id| id.as_ref().map_inner(|s| s.borrow()));
        self.declare_impl(diags, id, value)
    }

    pub fn declare_impl(&mut self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        let id = match id {
            Ok(id) => id,
            Err(e) => {
                self.content.any_id_err = Err(e);
                return;
            }
        };

        match self.content.values.entry(id.inner.to_owned()) {
            Entry::Occupied(mut entry) => {
                // already declared, report error
                let declared = entry.get_mut();

                // get all spans
                let mut spans = match declared {
                    DeclaredValue::Once { value: _, span } => vec![*span],
                    DeclaredValue::Multiple { spans, err: _ } => std::mem::take(spans),
                    DeclaredValue::Error(_) => return,
                    DeclaredValue::FailedCapture(_, _) => {
                        // TODO is this really not reachable in normal code?
                        let _ = diags
                            .report_error_internal(id.span, "declaring in scope that already has failed capture value");
                        return;
                    }
                };

                // report error
                // TODO this creates O(n^2) lines of errors, ideally we only want to report the final O(n) one
                let mut diag = DiagnosticError::new(
                    format!("identifier `{}` declared multiple times", id.inner),
                    id.span,
                    "declared again here",
                );
                for span in &spans {
                    diag = diag.add_info(*span, "previously declared here");
                }
                let err = diag.report(diags);

                // insert error value into scope to avoid downstream errors
                //   caused by only considering the first declared value
                spans.push(id.span);
                *declared = DeclaredValue::Multiple { spans, err };
            }
            Entry::Vacant(entry) => {
                entry.insert(DeclaredValue::Once { value, span: id.span });
            }
        }
    }

    pub fn declare_already_checked(&mut self, id: String, value: DeclaredValueSingle) {
        match self.content.values.entry(id.clone()) {
            Entry::Occupied(_) => panic!("identifier `{}` already declared in scope {:?}", id, self.span),
            Entry::Vacant(entry) => {
                let declared = match value {
                    DeclaredValueSingle::Value { value, span } => DeclaredValue::Once { value: Ok(value), span },
                    DeclaredValueSingle::FailedCapture(span, reason) => DeclaredValue::FailedCapture(span, reason),
                    DeclaredValueSingle::Error(err) => DeclaredValue::Error(err),
                };
                entry.insert(declared);
            }
        }
    }

    pub fn maybe_declare(
        &mut self,
        diags: &Diagnostics,
        id: DiagResult<MaybeIdentifier<Spanned<impl Borrow<str>>>>,
        entry: DiagResult<ScopedEntry>,
    ) {
        let id = match id {
            Ok(MaybeIdentifier::Dummy { span: _ }) => return,
            Ok(MaybeIdentifier::Identifier(id)) => Ok(id),
            Err(e) => Err(e),
        };
        self.declare(diags, id, entry);
    }

    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    pub fn find(&self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<ScopeFound> {
        self.find_impl(diags, id.inner, Some(id.span), self.span, true)
    }

    pub fn find_immediate_str(&self, diags: &Diagnostics, id: &str) -> DiagResult<ScopeFound> {
        self.find_impl(diags, id, None, self.span, false)
    }

    fn find_impl(
        &self,
        diags: &Diagnostics,
        id: &str,
        id_span: Option<Span>,
        initial_scope_span: Span,
        check_parents: bool,
    ) -> DiagResult<ScopeFound> {
        match self.try_find_impl(diags, id, id_span, initial_scope_span, check_parents)? {
            TryScopeFound::Found(found) => Ok(found),
            TryScopeFound::NotFoundAnyIdErr(any_id_err) => {
                // if any dynamic id declaration failed, propagate that error to supress superfluous errors
                any_id_err?;

                // report an error
                let info_parents = if check_parents { " and its parents" } else { "" };

                // TODO add fuzzy-matched suggestions as info
                // TODO simplify once we we always have a span, eg. from a top config file, commandline or python callsite
                let title = format!("undeclared identifier `{id}`");

                let diag = if let Some(id_span) = id_span {
                    DiagnosticError::new(title, id_span, "identifier not declared").add_info(
                        Span::empty_at(initial_scope_span.start()),
                        format!("searched in the scope starting here{info_parents}"),
                    )
                } else {
                    DiagnosticError::new(
                        title,
                        Span::empty_at(initial_scope_span.start()),
                        format!("searched in the scope starting here{info_parents}"),
                    )
                };
                Err(diag.report(diags))
            }
        }
    }

    fn try_find_impl(
        &self,
        diags: &Diagnostics,
        id: &str,
        id_span: Option<Span>,
        initial_scope_span: Span,
        check_parents: bool,
    ) -> DiagResult<TryScopeFound> {
        let mut curr = self;
        let mut any_id_err = Ok(());

        loop {
            if let Some(declared) = curr.content.values.get(id) {
                let (value, value_span) = match *declared {
                    DeclaredValue::Once { value, span } => (value?, span),
                    DeclaredValue::Multiple { spans: _, err } => return Err(err),
                    DeclaredValue::Error(err) => return Err(err),
                    DeclaredValue::FailedCapture(span, reason) => {
                        let reason_str = match reason {
                            FailedCaptureReason::NotCompile => "contains a non-compile-time value",
                            FailedCaptureReason::NotFullyInitialized => "was not fully initialized",
                        };

                        let id_span = id_span.ok_or_else(|| {
                            diags.report_error_internal(
                                initial_scope_span,
                                "tried to resolve span-less id in captured context",
                            )
                        })?;

                        return Err(DiagnosticError::new(
                            format!("failed to capture identifier because {reason_str}"),
                            id_span,
                            "used here",
                        )
                        .add_info(span, "value declared here")
                        .report(diags));
                    }
                };

                return Ok(TryScopeFound::Found(ScopeFound {
                    defining_span: value_span,
                    value,
                }));
            }

            any_id_err = any_id_err.and(curr.content.any_id_err);

            curr = if check_parents {
                if let ScopeParent::Some(parent) = curr.parent {
                    parent
                } else {
                    break;
                }
            } else {
                break;
            };
        }

        Ok(TryScopeFound::NotFoundAnyIdErr(any_id_err))
    }
}
