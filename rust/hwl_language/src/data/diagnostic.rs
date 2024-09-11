use crate::data::source::SourceDatabase;
use crate::syntax::ast::Identifier;
use crate::syntax::pos::{DifferentFile, Span};
use annotate_snippets::renderer::{AnsiColor, Color, Style};
use annotate_snippets::{Level, Renderer, Snippet};
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::cmp::min;

// TODO give this a better name to clarify that this means that the compiler gave up on this

/// Indicates that an error was already reported as a diagnostic.
///
/// Anything encountering this error should do one of, in order of preference:
/// * Continue on a best-effort basis,
///   allowing future errors that are definitely independent of the previous one to also be reported.
/// * Propagate this value,
///   blocking any downstream errors that are potentially just caused by the previous error
///   and which would just be noise.
///
/// This value is not publicly constructible, use [Diagnostics::report]. 
#[must_use]
#[derive(Debug, Copy, Clone)]
pub struct DiagnosticError(());

pub type DiagnosticResult<T> = Result<T, DiagnosticError>;
pub type DiagnosticResultPartial<T> = Result<T, (T, DiagnosticError)>;

pub struct Diagnostics {
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self { diagnostics: RefCell::new(vec![]) }
    }

    pub fn report_and_continue(&self, diag: Diagnostic) {
        self.diagnostics.borrow_mut().push(diag);
    }

    // TODO go through and try to avoid early-exits as much as possible
    pub fn report(&self, diag: Diagnostic) -> DiagnosticError {
        self.report_and_continue(diag);
        DiagnosticError(())
    }

    pub fn report_todo(&self, span: Span, feature: &str) -> DiagnosticError {
        self.report(Diagnostic::new_todo(span, feature))
    }

    pub fn finish(self) -> Vec<Diagnostic> {
        self.diagnostics.into_inner()
    }
}

// TODO separate errors and warnings
#[must_use]
#[derive(Debug)]
pub struct Diagnostic {
    pub title: String,
    pub snippets: Vec<(Span, Vec<Annotation>)>,
    pub footers: Vec<(Level, String)>,
}

#[derive(Debug, PartialEq)]
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
}

impl Default for DiagnosticStringSettings {
    fn default() -> Self {
        DiagnosticStringSettings {
            snippet_context_lines: 2,
            snippet_merge_max_distance: Some(3),
        }
    }
}

impl Diagnostic {
    pub fn new(title: impl Into<String>) -> DiagnosticBuilder {
        DiagnosticBuilder {
            diagnostic: Diagnostic {
                title: title.into(),
                snippets: vec![],
                footers: vec![],
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
        Diagnostic::new(format!("duplicate {:?}", kind))
            .snippet(span)
            .add_info(prev.span, "previously defined here")
            .add_error(curr.span, "defined for the second time here")
            .finish()
            .finish()
    }

    /// Utility diagnostic constructor for features that are not yet implemented.
    #[track_caller]
    pub fn new_todo(span: Span, feature: &str) -> Diagnostic {
        let message = format!("feature not yet implemented: '{}'", feature);
        let backtrace = Backtrace::force_capture();

        Diagnostic::new(&message)
            .add_error(span, "used here")
            .footer(Level::Info, backtrace.to_string())
            .finish()
    }

    pub fn to_string(self, database: &SourceDatabase, settings: DiagnosticStringSettings) -> String {
        let Self { title, snippets, footers } = self;

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
        assert!(!self.diagnostic.snippets.is_empty(), "Diagnostic without any snippets is not allowed");
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

    fn add_info(self, span: Span, label: impl Into<String>) -> Self {
        self.add(Level::Info, span, label)
    }
}
