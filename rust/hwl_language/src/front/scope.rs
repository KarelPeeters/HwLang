use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::function::FailedCaptureReason;
use crate::front::misc::ScopedEntry;
use crate::syntax::ast;
use crate::syntax::ast::Identifier;
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use indexmap::map::{Entry, IndexMap};
use std::fmt::Debug;

// TODO use string interning to avoid a bunch of string equality checks and hashing
#[derive(Debug)]
pub struct Scope<'p> {
    span: Span,
    parent: ScopeParent<'p>,
    content: ScopeContent,
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
    check: u64,
    values: IndexMap<String, DeclaredValue>,
    parent_check: Option<u64>,
}

// TODO simplify all of this: we might only only need to report errors on the first re-declaration,
//   which means we can remove that branch entirely
#[derive(Debug)]
enum DeclaredValue {
    Once {
        value: Result<ScopedEntry, ErrorGuaranteed>,
        span: Span,
    },
    Multiple {
        spans: Vec<Span>,
        err: ErrorGuaranteed,
    },
    FailedCapture(Span, FailedCaptureReason),
    Error(ErrorGuaranteed),
}

#[derive(Debug)]
pub enum DeclaredValueSingle<S = ScopedEntry> {
    Value { span: Span, value: S },
    FailedCapture(Span, FailedCaptureReason),
    Error(ErrorGuaranteed),
}

#[derive(Debug)]
pub struct ScopeFound<'s> {
    pub defining_span: Span,
    pub value: &'s ScopedEntry,
}

impl<'p> Scope<'p> {
    pub fn new_root(span: Span, file: FileId) -> Self {
        Scope {
            span,
            parent: ScopeParent::None(file),
            content: ScopeContent {
                check: rand::random(),
                values: IndexMap::new(),
                parent_check: None,
            },
        }
    }

    pub fn new_child(span: Span, parent: &'p Scope<'p>) -> Self {
        Scope {
            span,
            parent: ScopeParent::Some(parent),
            content: ScopeContent {
                check: rand::random(),
                values: IndexMap::new(),
                parent_check: Some(parent.content.check),
            },
        }
    }

    pub fn restore_child_from_content(span: Span, parent: &'p Scope<'p>, values: ScopeContent) -> Self {
        assert_eq!(Some(parent.content.check), values.parent_check);
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

    pub fn immediate_entries(&self) -> impl Iterator<Item = (&str, DeclaredValueSingle<&ScopedEntry>)> {
        self.content.values.iter().map(|(k, v)| {
            let v = match v {
                &DeclaredValue::Once { ref value, span } => match value {
                    Ok(value) => DeclaredValueSingle::Value { span, value },
                    &Err(e) => DeclaredValueSingle::Error(e),
                },
                &DeclaredValue::Multiple { spans: _, err } => DeclaredValueSingle::Error(err),
                &DeclaredValue::FailedCapture(span, reason) => DeclaredValueSingle::FailedCapture(span, reason),
                &DeclaredValue::Error(err) => DeclaredValueSingle::Error(err),
            };
            (k.as_str(), v)
        })
    }

    /// Declare a value in this scope.
    ///
    /// Allows shadowing identifiers in the parent scope, but not in the local scope.
    ///
    /// This function always appears to succeed, errors are instead reported as diags.
    /// This also tracks identifiers that have erroneously been declared multiple times,
    /// so that [Scope::find] can return an error for those cases.
    pub fn declare(&mut self, diags: &Diagnostics, id: &Identifier, value: Result<ScopedEntry, ErrorGuaranteed>) {
        if let Some(declared) = self.content.values.get_mut(&id.string) {
            // get all spans
            let mut spans = match declared {
                DeclaredValue::Once { value: _, span } => vec![*span],
                DeclaredValue::Multiple { spans, err: _ } => std::mem::take(spans),
                DeclaredValue::Error(_) => return,
                DeclaredValue::FailedCapture(_, _) => {
                    diags.report_internal_error(id.span, "declaring in scope that already has failed capture value");
                    return;
                }
            };

            // report error
            // TODO this creates O(n^2) lines of errors, ideally we only want to report the final O(n) one
            let mut diag = Diagnostic::new("identifier declared multiple times");
            for span in &spans {
                diag = diag.add_info(*span, "previously declared here");
            }
            let diag = diag.add_error(id.span, "declared again here").finish();
            let err = diags.report(diag);

            // insert error value into scope to avoid downstream errors
            //   caused by only considering the first declared value
            spans.push(id.span);
            *declared = DeclaredValue::Multiple { spans, err }
        } else {
            let declared = DeclaredValue::Once { value, span: id.span };
            self.content.values.insert_first(id.string.to_owned(), declared);
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
        id: ast::MaybeIdentifier<&Identifier>,
        entry: Result<ScopedEntry, ErrorGuaranteed>,
    ) {
        match id {
            ast::MaybeIdentifier::Identifier(id) => self.declare(diags, id, entry),
            ast::MaybeIdentifier::Dummy(_) => {}
        }
    }

    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    pub fn find<'s>(&'s self, diags: &Diagnostics, id: &Identifier) -> Result<ScopeFound<'s>, ErrorGuaranteed> {
        self.find_impl(diags, &id.string, Some(id.span), self.span, true)
    }

    pub fn find_immediate_str(&self, diags: &Diagnostics, id: &str) -> Result<ScopeFound, ErrorGuaranteed> {
        self.find_impl(diags, id, None, self.span, false)
    }

    fn find_impl<'s>(
        &'s self,
        diags: &Diagnostics,
        id: &str,
        id_span: Option<Span>,
        initial_scope_span: Span,
        check_parents: bool,
    ) -> Result<ScopeFound<'s>, ErrorGuaranteed> {
        if let Some(declared) = self.content.values.get(id) {
            // check declared exactly once
            let (value, value_span) = match *declared {
                DeclaredValue::Once { ref value, span } => (value.as_ref_ok()?, span),
                DeclaredValue::Multiple { spans: _, err } => return Err(err),
                DeclaredValue::Error(err) => return Err(err),
                DeclaredValue::FailedCapture(span, reason) => {
                    let reason_str = match reason {
                        FailedCaptureReason::NotCompile => "contained a non-compile-time value",
                        FailedCaptureReason::NotFullyInitialized => "was not fully initialized",
                    };
                    let err = diags.report_simple(
                        format!("failed to capture value because it {}", reason_str),
                        span,
                        "value set here",
                    );
                    return Err(err);
                }
            };

            return Ok(ScopeFound {
                defining_span: value_span,
                value,
            });
        }

        if check_parents {
            if let ScopeParent::Some(parent) = self.parent {
                return parent.find_impl(diags, id, id_span, initial_scope_span, check_parents);
            }
        }

        // TODO insert error entry to supress future errors that ask for the same identifier?
        let info_parents = if check_parents { " and its parents" } else { "" };

        // TODO add fuzzy-matched suggestions as info
        // TODO insert identifier into the current scope to suppress downstream errors
        // TODO simplify once we we always have a span, eg. from a top config file, commandline or python callsite
        let mut diag = Diagnostic::new(format!("undeclared identifier `{}`", id));
        if let Some(id_span) = id_span {
            diag = diag.add_error(id_span, "identifier not declared");
            diag = diag.add_info(
                Span::empty_at(initial_scope_span.start),
                format!("searched in the scope starting here{info_parents}"),
            );
        } else {
            diag = diag.add_error(
                Span::empty_at(initial_scope_span.start),
                format!("searched in the scope starting here{info_parents}"),
            );
        }

        Err(diags.report(diag.finish()))
    }
}
