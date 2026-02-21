use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, Diagnostics};
use crate::front::flow::{FailedCaptureReason, Variable};
use crate::front::signal::{Port, PortInterface, Register, Wire, WireInterface};
use crate::syntax::ast::MaybeIdentifier;
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::source::FileId;
use indexmap::map::{Entry, IndexMap};
use std::cell::RefCell;
use std::fmt::Debug;

#[derive(Debug)]
pub struct FileScope {
    content: ScopeContent,
}

#[derive(Debug)]
pub struct Scope<'p> {
    span: Span,
    parent: ScopeParent<'p>,
    content: RefCell<ScopeContent>,
}

#[derive(Debug, Copy, Clone)]
pub enum ScopeParent<'p> {
    File(&'p FileScope),
    Normal(&'p Scope<'p>),
}

// TODO use string interning to avoid a bunch of string equality checks and hashing
#[derive(Debug)]
pub struct ScopeContent {
    span: Span,
    values: IndexMap<String, DeclaredValue>,
    any_id_err: DiagResult,
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

impl FileScope {
    pub fn new(span: Span) -> Self {
        FileScope {
            content: ScopeContent::new(span),
        }
    }

    pub fn as_scope(&self) -> Scope<'_> {
        Scope::new(self.file_span(), ScopeParent::File(self))
    }

    pub fn new_child(&self, span: Span) -> Scope<'_> {
        Scope::new(span, ScopeParent::File(self))
    }

    pub fn file(&self) -> FileId {
        self.content.span.file
    }

    pub fn file_span(&self) -> Span {
        self.content.span
    }

    pub fn declare(&mut self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        self.content.declare(diags, id, value);
    }

    pub fn maybe_declare(
        &mut self,
        diags: &Diagnostics,
        id: DiagResult<MaybeIdentifier<Spanned<&str>>>,
        entry: DiagResult<ScopedEntry>,
    ) {
        self.content.maybe_declare(diags, id, entry)
    }

    pub fn declare_already_checked(&mut self, id: String, value: DeclaredValueSingle) {
        self.content.declare_already_checked(id, value);
    }

    pub fn find(&self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<ScopeFound> {
        self.content
            .find_step(diags, id)?
            .ok_or_else(|| error_not_found(self.content.span, id, false).report(diags))
    }

    pub fn for_each_immediate_entry(&self, f: impl FnMut(&str, DeclaredValueSingle<ScopedEntry>)) {
        self.content.for_each_immediate_entry(f);
    }

    pub fn has_immediate_entry(&self, id: &str) -> bool {
        self.content.has_immediate_entry(id)
    }
}

impl<'p> Scope<'p> {
    pub fn new_child(&'p self, span: Span) -> Self {
        Scope::new(span, ScopeParent::Normal(self))
    }

    fn new(span: Span, parent: ScopeParent<'p>) -> Self {
        Scope {
            span,
            parent,
            content: RefCell::new(ScopeContent::new(span)),
        }
    }

    pub fn restore_from_content(parent: ScopeParent<'p>, content: ScopeContent) -> Self {
        Scope {
            span: content.span,
            parent,
            content: RefCell::new(content),
        }
    }

    pub fn parent(&self) -> ScopeParent<'p> {
        self.parent
    }

    pub fn into_content(self) -> ScopeContent {
        self.content.into_inner()
    }

    pub fn for_each_immediate_entry(&self, f: impl FnMut(&str, DeclaredValueSingle<ScopedEntry>)) {
        self.content.borrow().for_each_immediate_entry(f);
    }

    /// Declare a value in this scope.
    ///
    /// Allows shadowing identifiers in the parent scope, but not in the local scope.
    ///
    /// This function always appears to succeed, errors are instead reported as diags.
    /// This also tracks identifiers that have erroneously been declared multiple times,
    /// so that [Scope::find] can return an error for those cases.
    pub fn declare(&mut self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        self.declare_impl(diags, id, value)
    }

    /// The same as [Self::declare], but does not require `&mut self`.
    ///
    /// This should only be used in cases where a declaration needs to happen in a borrowed parent scope.
    /// We don't want to use this as the default declare method, since usually that is an error.
    pub fn declare_non_mut(&self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        self.declare_impl(diags, id, value)
    }

    fn declare_impl(&self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        let id = match id {
            Ok(id) => id,
            Err(e) => {
                self.content.borrow_mut().any_id_err = Err(e);
                return;
            }
        };

        match self.content.borrow_mut().values.entry(id.inner.to_owned()) {
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
        self.content.borrow_mut().declare_already_checked(id, value);
    }

    pub fn maybe_declare(
        &mut self,
        diags: &Diagnostics,
        id: DiagResult<MaybeIdentifier<Spanned<&str>>>,
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
        self.find_impl(diags, id, true)
    }

    pub fn find_without_parents(&self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<ScopeFound> {
        self.find_impl(diags, id, false)
    }

    fn find_impl(&self, diags: &Diagnostics, id: Spanned<&str>, check_parents: bool) -> DiagResult<ScopeFound> {
        let mut curr = self;
        loop {
            let content = curr.content.borrow();
            if let Some(found) = content.find_step(diags, id)? {
                return Ok(found);
            }

            curr = if check_parents {
                match curr.parent {
                    ScopeParent::File(parent) => {
                        if let Some(found) = parent.content.find_step(diags, id)? {
                            return Ok(found);
                        }
                        break;
                    }
                    ScopeParent::Normal(parent) => parent,
                }
            } else {
                break;
            };
        }

        Err(error_not_found(self.span, id, check_parents).report(diags))
    }
}

impl ScopeContent {
    pub fn new(span: Span) -> ScopeContent {
        ScopeContent {
            span,
            values: IndexMap::new(),
            any_id_err: Ok(()),
        }
    }

    fn maybe_declare(
        &mut self,
        diags: &Diagnostics,
        id: DiagResult<MaybeIdentifier<Spanned<&str>>>,
        entry: DiagResult<ScopedEntry>,
    ) {
        let id = match id {
            Ok(MaybeIdentifier::Dummy { span: _ }) => return,
            Ok(MaybeIdentifier::Identifier(id)) => Ok(id),
            Err(e) => Err(e),
        };
        self.declare(diags, id, entry);
    }

    pub fn declare(&mut self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        self.declare_impl(diags, id, value)
    }

    fn declare_impl(&mut self, diags: &Diagnostics, id: DiagResult<Spanned<&str>>, value: DiagResult<ScopedEntry>) {
        let id = match id {
            Ok(id) => id,
            Err(e) => {
                self.any_id_err = Err(e);
                return;
            }
        };

        match self.values.entry(id.inner.to_owned()) {
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

    // TODO do we really need this?
    pub fn declare_already_checked(&mut self, id: String, value: DeclaredValueSingle) {
        match self.values.entry(id.clone()) {
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

    fn find_step(&self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<Option<ScopeFound>> {
        self.any_id_err?;

        if let Some(declared) = self.values.get(id.inner) {
            let (value, defining_span) = match *declared {
                DeclaredValue::Once { value, span } => (value?, span),
                DeclaredValue::Multiple { spans: _, err } => return Err(err),
                DeclaredValue::Error(err) => return Err(err),
                DeclaredValue::FailedCapture(span, reason) => {
                    let reason_str = match reason {
                        FailedCaptureReason::NotCompile => "contains a non-compile-time value",
                        FailedCaptureReason::NotFullyInitialized => "was not fully initialized",
                    };
                    return Err(DiagnosticError::new(
                        format!("failed to capture value because it {reason_str}"),
                        id.span,
                        "used here",
                    )
                    .add_info(span, "value declared here")
                    .report(diags));
                }
            };

            Ok(Some(ScopeFound { defining_span, value }))
        } else {
            Ok(None)
        }
    }

    pub fn for_each_immediate_entry(&self, mut f: impl FnMut(&str, DeclaredValueSingle<ScopedEntry>)) {
        for (k, v) in &self.values {
            let v = match *v {
                DeclaredValue::Once { value, span } => match value {
                    Ok(value) => DeclaredValueSingle::Value { span, value },
                    Err(e) => DeclaredValueSingle::Error(e),
                },
                DeclaredValue::Multiple { spans: _, err } => DeclaredValueSingle::Error(err),
                DeclaredValue::FailedCapture(span, reason) => DeclaredValueSingle::FailedCapture(span, reason),
                DeclaredValue::Error(err) => DeclaredValueSingle::Error(err),
            };
            f(k.as_str(), v)
        }
    }

    pub fn has_immediate_entry(&self, id: &str) -> bool {
        self.values.contains_key(id)
    }
}

fn error_not_found(initial_scope_span: Span, id: Spanned<&str>, check_parents: bool) -> DiagnosticError {
    // TODO add fuzzy-matched suggestions as info
    // TODO simplify once we we always have a span, eg. from a top config file, commandline or python callsite
    let title = format!("undeclared identifier `{}`", id.inner);
    let msg_searched = if check_parents {
        "searched in the scope starting here and its parents"
    } else {
        "searched in the scope starting"
    };
    DiagnosticError::new(title, id.span, "identifier not declared")
        .add_info(Span::empty_at(initial_scope_span.start()), msg_searched)
}
