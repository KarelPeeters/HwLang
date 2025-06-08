//! This module is strongly inspired by the Rust compiler,
//! see <https://rustc-dev-guide.rust-lang.org/diagnostics.html>.

use crate::syntax::pos::{DifferentFile, Span};
use crate::syntax::source::SourceDatabase;
use annotate_snippets::renderer::{AnsiColor, Color, Style};
use annotate_snippets::{Level, Renderer, Snippet};
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::cmp::{min, Ordering};

// TODO give this a better name to clarify that this means that the compiler gave up on this
//   and that the error has already been reported as a diagnostic.
//   The current name is copied from the rust compiler:
//

/// Indicates that an error was reported as a diagnostic.
///
/// Anything encountering this value should do one of, in order of preference:
/// * Continue on a best-effort basis,
///   allowing future errors that are definitely independent of the previous one to also be reported.
///   The compiler should make optimistic assumptions about whatever value is instead [DiagError].
/// * Propagate this value,
///   blocking any downstream errors that are potentially just caused by the previous error
///   and which would just be noise.
///
/// This value is not publicly constructible, use [Diagnostics::report].
///
/// This concept is the same as
/// [ErrorGuaranteed](https://rustc-dev-guide.rust-lang.org/diagnostics/error-guaranteed.html)
/// from the rust compiler.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct DiagError(());

pub type DiagResult<T> = Result<T, DiagError>;

#[macro_export]
macro_rules! impl_from_error_guaranteed {
    ($ty:ident) => {
        impl From<DiagError> for $ty {
            fn from(e: DiagError) -> Self {
                $ty::Error(e)
            }
        }
    };
}

#[must_use]
#[derive(Clone)]
pub struct Diagnostics {
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            diagnostics: RefCell::new(vec![]),
        }
    }

    // TODO go through and try to avoid early-exits as much as possible
    // TODO limit the number of diagnostics reported, eg. stop after 1k
    pub fn report(&self, diag: Diagnostic) -> DiagError {
        self.diagnostics.borrow_mut().push(diag);
        DiagError(())
    }

    // TODO only single string parameter, used for both title and label?
    pub fn report_simple(&self, title: impl Into<String>, span: Span, label: impl Into<String>) -> DiagError {
        self.report(Diagnostic::new_simple(title, span, label))
    }

    #[track_caller]
    pub fn report_todo(&self, span: Span, feature: impl Into<String>) -> DiagError {
        self.report(Diagnostic::new_todo(feature).add_error(span, "used here").finish())
    }

    // TODO rename to "report_bug"
    pub fn report_internal_error(&self, span: Span, reason: impl Into<String>) -> DiagError {
        self.report(Diagnostic::new_internal_error(span, reason))
    }

    // TODO sort diagnostics by location, for a better user experience?
    //   especially with the graph traversal stuff, the order can be very arbitrary
    pub fn finish(self) -> Vec<Diagnostic> {
        self.diagnostics.into_inner()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.borrow().len()
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
    diagnostic: Diagnostic,
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

    /// Whether to include a backtrace in todo and internal compiler error diagnostics.
    backtrace: bool,
}

impl Default for DiagnosticStringSettings {
    fn default() -> Self {
        DiagnosticStringSettings {
            snippet_context_lines: 2,
            snippet_merge_max_distance: Some(3),
            backtrace: false,
        }
    }
}

// TODO make it clear in which phase each diagnostic was reported: file loading, parsing, type checking, lowering
// TODO clarify constructor naming: start_ for builders and new_ for complete?
//   what is the point of complete constructors in the first place?
impl Diagnostic {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(title: impl Into<String>) -> DiagnosticBuilder {
        DiagnosticBuilder {
            diagnostic: Diagnostic {
                title: title.into(),
                snippets: vec![],
                footers: vec![],
                backtrace: None,
            },
        }
    }

    /// Utility diagnostic constructor for features that are not yet implemented.
    #[track_caller]
    pub fn new_todo(feature: impl Into<String>) -> DiagnosticBuilder {
        let message = format!("feature not yet implemented: '{}'", feature.into());
        let mut diag = Diagnostic::new(&message);
        diag.diagnostic.backtrace = Some(Backtrace::force_capture().to_string());
        diag
    }

    // TODO move this to a more logical place?
    /// Utility diagnostic constructor for a single error message with a single span.
    pub fn new_simple(title: impl Into<String>, span: Span, label: impl Into<String>) -> Diagnostic {
        Diagnostic::new(title).add_error(span, label).finish()
    }

    pub fn new_internal_error(span: Span, reason: impl Into<String>) -> Diagnostic {
        let mut diag =
            Diagnostic::new(format!("internal compiler error: '{}'", reason.into())).add_error(span, "caused here");
        diag.diagnostic.backtrace = Some(Backtrace::force_capture().to_string());
        diag.finish()
    }

    pub fn main_annotation(&self) -> Option<&Annotation> {
        // TODO make having at least a single annotation a type-system level requirement
        let mut top_annotation: Option<&Annotation> = None;
        for (_, annotations) in &self.snippets {
            for annotation in annotations {
                let is_better = match &top_annotation {
                    None => true,
                    // TODO better level comparison function
                    Some(prev) => compare_level(annotation.level, prev.level).is_gt(),
                };
                if is_better {
                    top_annotation = Some(annotation);
                }
            }
        }
        top_annotation
    }

    pub fn to_string(self, database: &SourceDatabase, settings: DiagnosticStringSettings) -> String {
        let Self {
            title,
            snippets,
            footers,
            backtrace,
        } = self;

        // TODO sort to ensure that the first snippet is one with the highest level,
        //   so it is always clickable

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
                    let distance = span_full.distance_lines(span_prev_full);

                    // check distance
                    let merge = match distance {
                        Ok(distance) => distance <= 2 * settings.snippet_context_lines + snippet_merge_max_distance,
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
            let file_info = &database[span.file];
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
                let Annotation {
                    span: span_annotation,
                    level,
                    label,
                } = annotation;

                let delta_start = span_annotation.start_byte - start_byte;
                let delta_end = span_annotation.end_byte - start_byte;
                let delta_span = delta_start..delta_end;

                snippet = snippet.annotation(level.span(delta_span).label(label));
            }

            message = message.snippet(snippet);
        }

        for &(level, ref footer) in &footers {
            message = message.footer(level.title(footer));
        }

        if let Some(backtrace) = &backtrace {
            if settings.backtrace {
                message = message.footer(Level::Info.title(backtrace));
            }
        }

        // format into string
        let renderer =
            Renderer::styled().emphasis(Style::new().bold().fg_color(Some(Color::Ansi(AnsiColor::BrightRed))));
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
        let Self {
            diag: mut builder,
            span,
            annotations,
        } = self;
        assert!(
            !annotations.is_empty(),
            "DiagnosticSnippetBuilder without any annotations is not allowed"
        );
        builder.diagnostic.snippets.push((span, annotations));
        builder
    }
}

impl DiagnosticAddable for DiagnosticSnippetBuilder {
    fn add(mut self, level: Level, span: Span, label: impl Into<String>) -> Self {
        assert!(
            self.span.contains_span(span),
            "DiagnosticSnippetBuilder labels must fall within snippet span"
        );
        self.annotations.push(Annotation {
            level,
            span,
            label: label.into(),
        });
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

pub fn compare_level(left: Level, right: Level) -> Ordering {
    (left as u8).cmp(&(right as u8)).reverse()
}
