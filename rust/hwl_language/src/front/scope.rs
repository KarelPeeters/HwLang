use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, Diagnostics};
use crate::front::flow::{Flow, Variable};
use crate::front::signal::{Interface, Signal};
use crate::front::value::{CompileValue, NotCompile, Value};
use crate::syntax::ast::MaybeIdentifier;
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::{Span, Spanned};
use crate::util::ResultExt;
use indexmap::map::{Entry, IndexMap};
use itertools::Either;
use std::cell::RefCell;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct FrozenScope {
    parent: Option<Arc<FrozenScope>>,
    content: ScopeContent,
}

#[derive(Debug)]
pub struct Scope<'p> {
    parent: ScopeParent<'p>,
    content: RefCell<ScopeContent>,
}

#[derive(Debug)]
pub enum ScopeParent<'p> {
    Frozen(Arc<FrozenScope>),
    Normal(&'p Scope<'p>),
}

// TODO use string interning to avoid a bunch of string equality checks and hashing
#[derive(Debug)]
pub struct ScopeContent {
    span: Span,
    self_value: Option<SelfValue>,
    values: IndexMap<String, DeclaredValue>,
}

// TODO do we actually still need clone bounds here?
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    /// Indirection though an item, the item should be evaluated.
    Item(AstRefItem),
    /// A named value: port, register, wire, variable.
    /// These are not fully evaluated immediately, they might be used symbolically
    ///   as assignment targets or in domain expressions.
    Named(NamedValue),
    /// Captured value, no longer directly connected to the original declaration.
    Captured(CapturedValue),
}

#[derive(Debug, Copy, Clone)]
pub enum NamedValue {
    Variable(Variable),
    Signal(Signal),
    Interface(Interface),
}

#[derive(Debug, Clone)]
pub struct SelfValue {
    span_decl: Span,
    value: Either<Value, CapturedValue>,
}

#[derive(Debug, Clone)]
pub struct CapturedValue {
    pub span_capture: Span,
    pub value: Result<Arc<CompileValue>, CaptureFailed>,
}

#[derive(Debug, Copy, Clone)]
pub enum CaptureFailed {
    NotCompile,
    NotFullyInitialized,
}

// TODO simplify all of this: we might only only need to report errors on the first re-declaration,
//   which means we can remove that branch entirely
// TODO we store the span twice, why?
#[derive(Debug)]
enum DeclaredValue {
    Once { value: DiagResult<ScopedEntry>, span: Span },
    Multiple { spans: Vec<Span>, err: DiagError },
    Error(DiagError),
}

#[derive(Debug, Copy, Clone)]
pub enum DeclaredValueSingle<S = ScopedEntry> {
    Value { span: Span, value: S },
    Error(DiagError),
}

#[derive(Debug)]
pub struct ScopeFound {
    pub span_decl: Span,
    pub value: ScopedEntry,
}

impl FrozenScope {
    pub fn new(span: Span) -> Self {
        FrozenScope {
            parent: None,
            content: ScopeContent::new(span),
        }
    }

    pub fn as_scope(self: Arc<FrozenScope>) -> Scope<'static> {
        Scope::new(self.content.span, ScopeParent::Frozen(self))
    }

    pub fn new_child(self: Arc<FrozenScope>, span: Span) -> Scope<'static> {
        Scope::new(span, ScopeParent::Frozen(self))
    }

    pub fn declare<'s>(
        &mut self,
        diags: &Diagnostics,
        id: impl Into<MaybeIdentifier<Spanned<&'s str>>>,
        value: DiagResult<ScopedEntry>,
    ) {
        self.content.declare(diags, id, value);
    }

    pub fn declare_already_checked(&mut self, id: String, value: DeclaredValueSingle) {
        self.content.declare_already_checked(id, value);
    }

    pub fn find(&self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<ScopeFound> {
        self.find_impl(diags, id, self.content.span)
    }

    fn find_impl(&self, diags: &Diagnostics, id: Spanned<&str>, start_span: Span) -> DiagResult<ScopeFound> {
        let mut curr = self;
        loop {
            if let Some(found) = curr.content.find_step(id)? {
                return Ok(found);
            }

            curr = match &curr.parent {
                Some(parent) => parent,
                None => break,
            }
        }

        Err(error_not_found(start_span, id).report(diags))
    }

    pub fn for_each_immediate_entry(&self, f: impl FnMut(&str, DeclaredValueSingle<&ScopedEntry>)) {
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
            parent,
            content: RefCell::new(ScopeContent::new(span)),
        }
    }

    pub fn restore_from_content(parent: ScopeParent<'p>, content: ScopeContent) -> Self {
        Scope {
            parent,
            content: RefCell::new(content),
        }
    }

    pub fn into_content(self) -> ScopeContent {
        self.content.into_inner()
    }

    pub fn for_each_immediate_entry(&self, f: impl FnMut(&str, DeclaredValueSingle<&ScopedEntry>)) {
        self.content.borrow().for_each_immediate_entry(f);
    }

    pub fn set_self_value(&self, diags: &Diagnostics, value: Spanned<Value>) -> DiagResult {
        self.content.borrow_mut().set_self_value(diags, value)
    }

    /// Declare a value in this scope.
    ///
    /// Allows shadowing identifiers in the parent scope, but not in the local scope.
    ///
    /// This function always appears to succeed, errors are instead reported as diags.
    /// This also tracks identifiers that have erroneously been declared multiple times,
    /// so that [Scope::find] can return an error for those cases.
    pub fn declare<'s>(
        &mut self,
        diags: &Diagnostics,
        id: impl Into<MaybeIdentifier<Spanned<&'s str>>>,
        value: DiagResult<ScopedEntry>,
    ) {
        self.content.get_mut().declare(diags, id, value)
    }

    /// The same as [Self::declare], but does not require `&mut self`.
    ///
    /// This should only be used in cases where a declaration needs to happen in a borrowed parent scope.
    /// We don't want to use this as the default declare method, since usually that is an error.
    pub fn declare_non_mut<'s>(
        &self,
        diags: &Diagnostics,
        id: impl Into<MaybeIdentifier<Spanned<&'s str>>>,
        value: DiagResult<ScopedEntry>,
    ) {
        self.content.borrow_mut().declare(diags, id, value)
    }

    pub fn declare_already_checked(&mut self, id: String, value: DeclaredValueSingle) {
        self.content.borrow_mut().declare_already_checked(id, value);
    }

    pub fn find_self_value(&self, diags: &Diagnostics, span_use: Span) -> DiagResult<Value> {
        let map_found = |found: &SelfValue| match &found.value {
            Either::Left(v) => Ok(v.clone()),
            Either::Right(v) => match &v.value {
                Ok(v) => Ok(Value::from(v.as_ref().clone())),
                &Err(e) => Err(e.to_diag_error(found.span_decl, v.span_capture, span_use).report(diags)),
            },
        };

        let mut curr = self;
        loop {
            let content = curr.content.borrow();
            if let Some(found) = &content.self_value {
                return map_found(found);
            }
            curr = match &curr.parent {
                ScopeParent::Normal(parent) => parent,
                ScopeParent::Frozen(parent) => {
                    let mut curr = parent;
                    loop {
                        if let Some(found) = &curr.content.self_value {
                            return map_found(found);
                        }
                        curr = match &curr.parent {
                            None => {
                                return Err(DiagnosticError::new(
                                    "self is not bound in this scope",
                                    span_use,
                                    "tried to use self here",
                                )
                                .report(diags));
                            }
                            Some(parent) => parent,
                        }
                    }
                }
            };
        }
    }

    /// Find the given identifier in this scope.
    /// Walks up into the parent scopes until a scope without a parent is found,
    /// then looks in the `root` scope. If no value is found returns `Err`.
    pub fn find(&self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<ScopeFound> {
        let mut curr = self;
        loop {
            let content = curr.content.borrow();
            if let Some(found) = content.find_step(id)? {
                return Ok(found);
            }
            curr = match &curr.parent {
                ScopeParent::Normal(parent) => parent,
                ScopeParent::Frozen(parent) => {
                    return parent.find_impl(diags, id, self.content.borrow().span);
                }
            };
        }
    }

    pub fn try_find_for_diagnostic(&self, id: &str) -> DiagResult<Option<Span>> {
        let mut curr = self;
        loop {
            if let Some(span) = curr.content.borrow().try_find_for_diagnostic(id)? {
                return Ok(Some(span));
            }

            curr = match &curr.parent {
                ScopeParent::Normal(parent) => parent,
                ScopeParent::Frozen(parent) => {
                    let mut curr = parent;
                    loop {
                        if let Some(span) = curr.content.try_find_for_diagnostic(id)? {
                            return Ok(Some(span));
                        }

                        curr = match &curr.parent {
                            Some(parent) => parent,
                            None => return Ok(None),
                        }
                    }
                }
            }
        }
    }

    pub fn capture(&self, flow: &impl Flow, span_capture: Span) -> FrozenScope {
        // walk up scopes, starting from the current scope up to the root
        //   try to capture all values that have not yet been shadowed by a child scope
        let mut captured_self_value: Option<SelfValue> = None;
        let mut captured_values: IndexMap<String, DeclaredValue> = IndexMap::new();

        let mut curr = self;
        let final_parent = loop {
            let curr_content = curr.content.borrow();

            // capture self, skip self already captured by child scopes
            if captured_self_value.is_none() {
                if let Some(self_value) = curr_content.self_value.as_ref() {
                    let value = match &self_value.value {
                        Either::Left(value) => match CompileValue::try_from(value) {
                            Ok(v_inner) => Ok(Arc::new(v_inner)),
                            Err(NotCompile) => Err(CaptureFailed::NotCompile),
                        },
                        Either::Right(value) => value.value.clone(),
                    };

                    captured_self_value = Some(SelfValue {
                        span_decl: self_value.span_decl,
                        value: Either::Right(CapturedValue { span_capture, value }),
                    });
                }
            }

            // capture normal entries
            curr_content.for_each_immediate_entry(|id, value| {
                // skip entries already captured by child scopes
                let child_values_entry = match captured_values.entry(id.to_owned()) {
                    Entry::Occupied(_) => return,
                    Entry::Vacant(child_values_entry) => child_values_entry,
                };

                let captured = match value {
                    DeclaredValueSingle::Value { span, value } => {
                        let captured_entry = match value {
                            &ScopedEntry::Item(item) => Ok(ScopedEntry::Item(item)),
                            &ScopedEntry::Named(named) => match named {
                                NamedValue::Variable(var) => {
                                    flow.var_capture(Spanned::new(span_capture, var)).map(|value| {
                                        ScopedEntry::Captured(CapturedValue {
                                            span_capture,
                                            value: value.map(Arc::new),
                                        })
                                    })
                                }
                                NamedValue::Signal(_) | NamedValue::Interface(_) => {
                                    Ok(ScopedEntry::Captured(CapturedValue {
                                        span_capture,
                                        value: Err(CaptureFailed::NotCompile),
                                    }))
                                }
                            },
                            // we're re-capturing, discard the original spans
                            ScopedEntry::Captured(cap) => Ok(ScopedEntry::Captured(CapturedValue {
                                span_capture,
                                value: cap.value.clone(),
                            })),
                        };

                        DeclaredValue::Once {
                            span,
                            value: captured_entry,
                        }
                    }
                    DeclaredValueSingle::Error(e) => DeclaredValue::Error(e),
                };

                child_values_entry.insert(captured);
            });

            curr = match &curr.parent {
                ScopeParent::Frozen(parent) => break parent,
                ScopeParent::Normal(parent) => parent,
            };
        };

        FrozenScope {
            parent: Some(Arc::clone(final_parent)),
            content: ScopeContent {
                span: self.content.borrow().span,
                self_value: captured_self_value,
                values: captured_values,
            },
        }
    }
}

impl ScopeContent {
    pub fn new(span: Span) -> ScopeContent {
        ScopeContent {
            span,
            self_value: None,
            values: IndexMap::new(),
        }
    }

    fn set_self_value(&mut self, diags: &Diagnostics, value: Spanned<Value>) -> DiagResult {
        if self.self_value.is_some() {
            let e = diags.report_error_internal(value.span, "cannot set self value multiple times in the same scope");
            return Err(e);
        }

        self.self_value = Some(SelfValue {
            span_decl: value.span,
            value: Either::Left(value.inner),
        });
        Ok(())
    }

    pub fn declare<'s>(
        &mut self,
        diags: &Diagnostics,
        id: impl Into<MaybeIdentifier<Spanned<&'s str>>>,
        value: DiagResult<ScopedEntry>,
    ) {
        match id.into() {
            MaybeIdentifier::Dummy { span: _ } => {}
            MaybeIdentifier::Identifier(id) => self.declare_impl(diags, id, value),
        }
    }

    fn declare_impl(&mut self, diags: &Diagnostics, id: Spanned<&str>, value: DiagResult<ScopedEntry>) {
        match self.values.entry(id.inner.to_owned()) {
            Entry::Occupied(mut entry) => {
                // already declared, report error
                let declared = entry.get_mut();

                // get all spans
                let mut spans = match declared {
                    DeclaredValue::Once { value: _, span } => vec![*span],
                    DeclaredValue::Multiple { spans, err: _ } => std::mem::take(spans),
                    DeclaredValue::Error(_) => return,
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
                    DeclaredValueSingle::Error(err) => DeclaredValue::Error(err),
                };
                entry.insert(declared);
            }
        }
    }

    fn find_step(&self, id: Spanned<&str>) -> DiagResult<Option<ScopeFound>> {
        if let Some(declared) = self.values.get(id.inner) {
            let (value, span_decl) = match *declared {
                DeclaredValue::Once { ref value, span } => (value.as_ref_ok()?, span),
                DeclaredValue::Multiple { spans: _, err } => return Err(err),
                DeclaredValue::Error(err) => return Err(err),
            };

            Ok(Some(ScopeFound {
                span_decl,
                value: value.clone(),
            }))
        } else {
            Ok(None)
        }
    }

    fn try_find_for_diagnostic(&self, id: &str) -> DiagResult<Option<Span>> {
        let result = if let Some(value) = self.values.get(id) {
            match value {
                &DeclaredValue::Once { value: _, span } => Some(span),
                DeclaredValue::Multiple { spans, err: _ } => Some(spans[0]),
                &DeclaredValue::Error(e) => return Err(e),
            }
        } else {
            None
        };
        Ok(result)
    }

    pub fn for_each_immediate_entry(&self, mut f: impl FnMut(&str, DeclaredValueSingle<&ScopedEntry>)) {
        for (k, v) in &self.values {
            let v = match *v {
                DeclaredValue::Once { ref value, span } => match value {
                    Ok(value) => DeclaredValueSingle::Value { span, value },
                    &Err(e) => DeclaredValueSingle::Error(e),
                },
                DeclaredValue::Multiple { spans: _, err } => DeclaredValueSingle::Error(err),
                DeclaredValue::Error(err) => DeclaredValueSingle::Error(err),
            };
            f(k.as_str(), v)
        }
    }

    pub fn has_immediate_entry(&self, id: &str) -> bool {
        self.values.contains_key(id)
    }
}

impl<S: Clone> DeclaredValueSingle<&S> {
    pub fn cloned(&self) -> DeclaredValueSingle<S> {
        match *self {
            DeclaredValueSingle::Value { span, value } => {
                let value = value.clone();
                DeclaredValueSingle::Value { span, value }
            }
            DeclaredValueSingle::Error(e) => DeclaredValueSingle::Error(e),
        }
    }
}

fn error_not_found(initial_scope_span: Span, id: Spanned<&str>) -> DiagnosticError {
    // TODO add fuzzy-matched suggestions as info
    // TODO simplify once we we always have a span, eg. from a top config file, commandline or python callsite
    let title = format!("undeclared identifier `{}`", id.inner);
    DiagnosticError::new(title, id.span, "identifier not declared").add_info(
        Span::empty_at(initial_scope_span.start()),
        "searched in the scope starting here and its parents",
    )
}

impl CaptureFailed {
    pub fn to_diag_error(self, span_decl: Span, span_capture: Span, span_use: Span) -> DiagnosticError {
        let reason = match self {
            CaptureFailed::NotCompile => "is not a compile-time value",
            CaptureFailed::NotFullyInitialized => "was not fully initialized",
        };
        DiagnosticError::new(
            format!("cannot access captured value because it {reason}"),
            span_use,
            "trying to access captured value here",
        )
        .add_info(span_decl, "value declared here")
        .add_info(span_capture, "value captured here")
    }
}
