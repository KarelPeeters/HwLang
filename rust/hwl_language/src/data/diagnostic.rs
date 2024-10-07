//! This module is strongly inspired by the Rust compiler,
//! see <https://rustc-dev-guide.rust-lang.org/diagnostics.html>.

use crate::data::source::SourceDatabase;
use crate::syntax::ast::Identifier;
use crate::syntax::pos::{DifferentFile, Span};
use annotate_snippets::renderer::{AnsiColor, Color, Style};
use annotate_snippets::{Level, Renderer, Snippet};
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::cmp::min;

// TODO give this a better name to clarify that this means that the compiler gave up on this

/// Indicates that an error was already reported.
///
/// Anything encountering this value should do one of, in order of preference:
/// * Continue on a best-effort basis,
///   allowing future errors that are definitely independent of the previous one to also be reported.
///   The compiler should make optimistic assumptions about whatever value is instead [ErrorGuaranteed].
/// * Propagate this value,
///   blocking any downstream errors that are potentially just caused by the previous error
///   and which would just be noise.
///
/// This value is not publicly constructible, use [Diagnostics::report].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ErrorGuaranteed(());

#[must_use]
pub struct Diagnostics {
    handler: Option<Box<dyn Fn(&Diagnostic)>>,
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::new_with_handler(None)
    }

    pub fn new_with_handler(handler: Option<Box<dyn Fn(&Diagnostic)>>) -> Self {
        Self {
            handler,
            diagnostics: RefCell::new(vec![]),
        }
    }

    // TODO go through and try to avoid early-exits as much as possible
    pub fn report(&self, diag: Diagnostic) -> ErrorGuaranteed {
        if let Some(handler) = &self.handler {
            handler(&diag);
        }

        self.diagnostics.borrow_mut().push(diag);
        ErrorGuaranteed(())
    }

    pub fn report_simple(&self, title: impl Into<String>, span: Span, label: impl Into<String>) -> ErrorGuaranteed {
        self.report(Diagnostic::new_simple(title, span, label))
    }

    pub fn report_todo(&self, span: Span, feature: impl Into<String>) -> ErrorGuaranteed {
        self.report(Diagnostic::new_todo(span, feature))
    }

    // TODO rename to "report_bug"
    pub fn report_internal_error(&self, span: Span, reason: impl Into<String>) -> ErrorGuaranteed {
        self.report(Diagnostic::new_internal_error(span, reason))
    }

    // TODO sort diagnostics by location, for a better user experience?
    //   especially with the graph traversal stuff, the order can be very arbitrary
    pub fn finish(self) -> Vec<Diagnostic> {
        self.diagnostics.into_inner()
    }
}

// TODO separate errors and warnings
#[must_use]
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub title: String,
    // TODO use Spanned here
    pub snippets: Vec<(Span, Vec<Annotation>)>,
    pub footers: Vec<(Level, String)>,
    pub backtrace: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub level: Level,
    pub span: Span,
    pub label: String,
}

#[must_use]
pub struct DiagnosticBuilder {
    diagnostic: Diagnostic
}

#[must_use]
pub struct DiagnosticSnippetBuilder {
    diag: DiagnosticBuilder,
    span: Span,
    annotations: Vec<Annotation>,
}

#[derive(Debug, Copy, Clone)]
pub struct DiagnosticStringSettings {
    /// The number of additional lines to show before and after each snippet range.
    snippet_context_lines: usize,

    /// The maximum distance between two snippets to merge them into one.
    /// This distance is measured after the context lines have already been added.
    /// If `None`, no merging is done.
    snippet_merge_max_distance: Option<usize>,

    /// Whether to include a backtrace in "todo" diagnostics.
    todo_backtrace: bool,
}

impl Default for DiagnosticStringSettings {
    fn default() -> Self {
        DiagnosticStringSettings {
            snippet_context_lines: 2,
            snippet_merge_max_distance: Some(3),
            todo_backtrace: false,
        }
    }
}

// TODO make it clear in which phase each diagnostic is reported: file loading, parsing, type checking, lowering
impl Diagnostic {
    pub fn new(title: impl Into<String>) -> DiagnosticBuilder {
        DiagnosticBuilder {
            diagnostic: Diagnostic {
                title: title.into(),
                snippets: vec![],
                footers: vec![],
                backtrace: None,
            }
        }
    }

    // TODO move this to a more logical place?
    /// Utility diagnostic constructor for a single error message with a single span.
    pub fn new_simple(title: impl Into<String>, span: Span, label: impl Into<String>) -> Diagnostic {
        Diagnostic::new(title)
            .add_error(span, label)
            .finish()
    }

    /// Utility diagnostic constructor for a duplicate identifier definition.
    pub fn new_defined_twice(kind: &str, span: Span, prev: &Identifier, curr: &Identifier) -> Diagnostic {
        Diagnostic::new(format!("duplicate {}", kind))
            .snippet(span)
            .add_info(prev.span, "previously defined here")
            .add_error(curr.span, "defined for the second time here")
            .finish()
            .finish()
    }

    /// Utility diagnostic constructor for features that are not yet implemented.
    #[track_caller]
    pub fn new_todo(span: Span, feature: impl Into<String>) -> Diagnostic {
        let message = format!("feature not yet implemented: '{}'", feature.into());
        let backtrace = Backtrace::force_capture();

        let mut diag = Diagnostic::new(&message)
            .add_error(span, "used here")
            .finish();
        diag.backtrace = Some(backtrace.to_string());
        diag
    }

    pub fn new_internal_error(span: Span, reason: impl Into<String>) -> Diagnostic {
        Diagnostic::new(format!("internal compiler error: '{}'", reason.into()))
            .add_error(span, "caused here")
            .finish()
    }

    pub fn to_string(self, database: &SourceDatabase, settings: DiagnosticStringSettings) -> String {
        let Self { title, snippets, footers, backtrace } = self;

        // combine snippets that are close together
        let snippets_merged = if let Some(snippet_merge_max_distance) = settings.snippet_merge_max_distance {
            // TODO fix O(n^2) complexity
            let mut snippets_merged: Vec<(Span, Vec<Annotation>)> = vec![];

            for (span, mut annotations) in snippets {
                // try merging with previous snippet
                let mut merged = false;
                for (span_prev, ref mut annotations_prev) in &mut snippets_merged {
                    // calculate distance
                    let span_full = database.expand_span(span);
                    let span_prev_full = database.expand_span(*span_prev);
                    let distance = span_full
                        .distance_lines(span_prev_full);

                    // check distance
                    let merge = match distance {
                        Ok(distance) =>
                            distance <= 2 * settings.snippet_context_lines + snippet_merge_max_distance,
                        Err(DifferentFile) => false,
                    };

                    // merge
                    if merge {
                        *span_prev = span_prev.join(span);
                        annotations_prev.append(&mut annotations);
                        merged = true;
                        break;
                    }
                }

                // failed to merge, just keep the new snippet
                if !merged {
                    snippets_merged.push((span, annotations));
                }
            }

            snippets_merged
        } else {
            snippets
        };

        // create final message
        let mut message = Level::Error.title(&title);

        for &(span, ref annotations) in &snippets_merged {
            let file_info = &database[span.start.file];
            let offsets = &file_info.offsets;

            // select lines and convert to bytes
            let span_snippet = offsets.expand_span(span);
            let start_line_0 = span_snippet.start.line_0.saturating_sub(settings.snippet_context_lines);
            let end_line_0 = min(
                span_snippet.end.line_0 + settings.snippet_context_lines,
                offsets.line_count() - 1,
            );
            let start_byte = offsets.line_start(start_line_0);
            let end_byte = offsets.line_end(end_line_0, false);
            let source = &file_info.source[start_byte..end_byte];

            // create snippet
            let mut snippet = Snippet::source(source)
                .origin(&file_info.path_raw)
                .line_start(start_line_0 + 1);
            for annotation in annotations {
                let Annotation { span: span_annotation, level, label } = annotation;

                let delta_start = span_annotation.start.byte - start_byte;
                let delta_end = span_annotation.end.byte - start_byte;
                let delta_span = delta_start..delta_end;

                snippet = snippet.annotation(level.span(delta_span).label(label));
            }

            message = message.snippet(snippet);
        }

        for &(level, ref footer) in &footers {
            message = message.footer(level.title(footer));
        }

        if let Some(backtrace) = &backtrace {
            if settings.todo_backtrace {
                message = message.footer(Level::Info.title(backtrace));
            }
        }

        // format into string
        let renderer = Renderer::styled()
            .emphasis(Style::new().bold().fg_color(Some(Color::Ansi(AnsiColor::BrightRed))));
        let render = renderer.render(message);
        render.to_string()
    }
}

impl DiagnosticBuilder {
    pub fn snippet(self, span: Span) -> DiagnosticSnippetBuilder {
        DiagnosticSnippetBuilder {
            diag: self,
            span,
            annotations: vec![],
        }
    }

    pub fn footer(mut self, level: Level, footer: impl Into<String>) -> Self {
        self.diagnostic.footers.push((level, footer.into()));
        self
    }

    pub fn finish(self) -> Diagnostic {
        self.diagnostic
    }
}

impl DiagnosticAddable for DiagnosticBuilder {
    fn add(self, level: Level, span: Span, label: impl Into<String>) -> Self {
        self.snippet(span).add(level, span, label).finish()
    }
}

impl DiagnosticSnippetBuilder {
    pub fn finish(self) -> DiagnosticBuilder {
        let Self { diag: mut builder, span, annotations } = self;
        assert!(!annotations.is_empty(), "DiagnosticSnippetBuilder without any annotations is not allowed");
        builder.diagnostic.snippets.push((span, annotations));
        builder
    }
}

impl DiagnosticAddable for DiagnosticSnippetBuilder {
    fn add(mut self, level: Level, span: Span, label: impl Into<String>) -> Self {
        assert!(self.span.contains(span), "DiagnosticSnippetBuilder labels must fall within snippet span");
        self.annotations.push(Annotation { level, span, label: label.into() });
        self
    }
}

pub trait DiagnosticAddable: Sized {
    fn add(self, level: Level, span: Span, label: impl Into<String>) -> Self;

    fn add_error(self, span: Span, label: impl Into<String>) -> Self {
        self.add(Level::Error, span, label)
    }

    fn add_error_maybe(self, span: Option<Span>, label: impl Into<String>) -> Self {
        if let Some(span) = span {
            self.add_error(span, label)
        } else {
            self
        }
    }

    fn add_info(self, span: Span, label: impl Into<String>) -> Self {
        self.add(Level::Info, span, label)
    }

    fn add_info_maybe(self, span: Option<Span>, label: impl Into<String>) -> Self {
        if let Some(span) = span {
            self.add_info(span, label)
        } else {
            self
        }
    }
}
