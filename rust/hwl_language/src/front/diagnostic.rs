//! This module is strongly inspired by the Rust compiler,
//! see <https://rustc-dev-guide.rust-lang.org/diagnostics.html>.

use crate::syntax::pos::{DifferentFile, Span};
use crate::syntax::source::SourceDatabase;
use crate::util::data::NonEmptyVec;
use annotate_snippets::renderer::{AnsiColor, Color, Style};
use annotate_snippets::{Level, Renderer, Snippet};
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::cmp::min;

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
#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct DiagError(());

pub type DiagResult<T = ()> = Result<T, DiagError>;

#[must_use]
#[derive(Debug, Clone)]
pub struct Diagnostics {
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            diagnostics: RefCell::new(vec![]),
        }
    }

    pub fn push(&self, diag: Diagnostic) {
        self.diagnostics.borrow_mut().push(diag);
    }

    pub fn report_error_simple(&self, title: impl Into<String>, span: Span, label: impl Into<String>) -> DiagError {
        DiagnosticError::new(title, span, label).report(self)
    }

    #[track_caller]
    pub fn report_error_todo(&self, span: Span, feature: impl AsRef<str>) -> DiagError {
        DiagnosticError::new_todo(feature, span).report(self)
    }

    pub fn report_error_internal(&self, span: Span, reason: impl AsRef<str>) -> DiagError {
        DiagnosticError::new_internal_compiler_error(reason, span).report(self)
    }

    pub fn finish(self) -> Vec<Diagnostic> {
        self.diagnostics.into_inner()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.borrow().len()
    }
}

impl DiagError {
    /// Create a [DiagError], promising that an error has already been reported.
    /// This is very rarely useful outside the diagnostics module itself to statically initialize constants.
    pub const fn promise_error_has_been_reported() -> DiagError {
        DiagError(())
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub content: DiagnosticContent,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum DiagnosticLevel {
    Error,
    Warning,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FooterKind {
    Info,
    Hint,
}

#[must_use]
#[derive(Debug, Clone)]
pub struct DiagnosticError {
    content: DiagnosticContent,
}

#[must_use]
#[derive(Debug, Clone)]
pub struct DiagnosticWarning {
    content: DiagnosticContent,
}

#[derive(Debug, Clone)]
pub struct DiagnosticContent {
    pub title: String,
    pub messages: NonEmptyVec<(Span, String)>,
    pub infos: Vec<(Span, String)>,
    pub footers: Vec<(FooterKind, String)>,
    pub backtrace: Option<String>,
}

impl DiagnosticError {
    pub fn new(title: impl Into<String>, span: Span, message: impl Into<String>) -> Self {
        let content = DiagnosticContent::new(title, span, message);
        Self { content }
    }

    pub fn new_multiple(title: impl Into<String>, messages: NonEmptyVec<(Span, String)>) -> Self {
        let content = DiagnosticContent::new_multiple(title, messages);
        Self { content }
    }

    pub fn new_todo(feature: impl AsRef<str>, span: Span) -> Self {
        let message = format!("feature not yet implemented: {}", feature.as_ref());
        Self::new(message, span, "used here")
    }

    #[track_caller]
    pub fn new_internal_compiler_error(reason: impl AsRef<str>, span: Span) -> Self {
        let message = format!("internal compiler error: {}", reason.as_ref());
        let mut diag = Self::new(message, span, "triggered here");
        diag.content.backtrace = Some(Backtrace::force_capture().to_string());
        diag
    }

    pub fn add_info(mut self, span: Span, info: impl Into<String>) -> Self {
        self.content.add_info(span, info.into());
        self
    }

    pub fn add_footer(mut self, kind: FooterKind, message: impl Into<String>) -> Self {
        self.content.add_footer(kind, message);
        self
    }

    pub fn report(self, diagnostics: &Diagnostics) -> DiagError {
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Error,
            content: self.content,
        });
        DiagError::promise_error_has_been_reported()
    }
}

impl DiagnosticWarning {
    pub fn new(title: impl Into<String>, span: Span, message: impl Into<String>) -> Self {
        let content = DiagnosticContent::new(title, span, message);
        Self { content }
    }

    pub fn new_multiple(title: impl Into<String>, messages: NonEmptyVec<(Span, String)>) -> Self {
        let content = DiagnosticContent::new_multiple(title, messages);
        Self { content }
    }

    pub fn add_info(mut self, span: Span, info: impl Into<String>) -> Self {
        self.content.add_info(span, info.into());
        self
    }

    pub fn add_footer(mut self, kind: FooterKind, message: impl Into<String>) -> Self {
        self.content.add_footer(kind, message);
        self
    }

    pub fn report(self, diagnostics: &Diagnostics) {
        diagnostics.push(Diagnostic {
            level: DiagnosticLevel::Warning,
            content: self.content,
        });
    }
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

    /// Whether to use ANSI color codes in the output.
    ansi_color: bool,
}

impl DiagnosticStringSettings {
    pub fn default(ansi_color: bool) -> Self {
        DiagnosticStringSettings {
            snippet_context_lines: 2,
            snippet_merge_max_distance: Some(3),
            backtrace: false,
            ansi_color,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Annotation {
    level: Level,
    span: Span,
    label: String,
}

impl Diagnostic {
    pub fn to_string(self, database: &SourceDatabase, settings: DiagnosticStringSettings) -> String {
        let Diagnostic { level, content } = self;
        content.to_string(database, settings, level)
    }

    pub fn sort_key(&self) -> impl Ord + '_ {
        // TODO expand until it contains all info, guaranteeing a single sort order
        (self.content.messages.first().0, self.level)
    }
}

impl DiagnosticContent {
    fn new(title: impl Into<String>, span: Span, message: impl Into<String>) -> Self {
        Self::new_multiple(title, NonEmptyVec::new_single((span, message.into())))
    }

    fn new_multiple(title: impl Into<String>, messages: NonEmptyVec<(Span, String)>) -> Self {
        DiagnosticContent {
            title: title.into(),
            messages,
            infos: vec![],
            footers: vec![],
            backtrace: None,
        }
    }

    fn add_info(&mut self, span: Span, info: impl Into<String>) {
        self.infos.push((span, info.into()));
    }

    fn add_footer(&mut self, kind: FooterKind, message: impl Into<String>) {
        self.footers.push((kind, message.into()));
    }

    fn to_string(
        self,
        database: &SourceDatabase,
        settings: DiagnosticStringSettings,
        level: DiagnosticLevel,
    ) -> String {
        let Self {
            title,
            messages,
            infos,
            footers,
            backtrace,
        } = self;

        let top_level_mapped = match level {
            DiagnosticLevel::Error => Level::Error,
            DiagnosticLevel::Warning => Level::Warning,
        };

        // convert to snippets
        let snippets = {
            let mut snippets = Vec::with_capacity(messages.len() + infos.len());

            for (top_span, top_message) in messages {
                let top_annotation = Annotation {
                    level: top_level_mapped,
                    span: top_span,
                    label: top_message,
                };
                snippets.push((top_span, vec![top_annotation]));
            }

            for (info_span, info_message) in infos {
                let info_annotation = Annotation {
                    level: Level::Info,
                    span: info_span,
                    label: info_message,
                };
                snippets.push((info_span, vec![info_annotation]));
            }

            snippets
        };

        // combine snippets that are close together
        let snippets_merged = if let Some(snippet_merge_max_distance) = settings.snippet_merge_max_distance {
            // TODO fix O(n^2) complexity
            let mut snippets_merged: Vec<(Span, Vec<Annotation>)> = vec![];

            for (span, mut annotations) in snippets {
                // try merging with previous snippet
                let mut merged = false;
                for (span_prev, annotations_prev) in &mut snippets_merged {
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
        let mut message = top_level_mapped.title(&title);

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
            let content = &file_info.content[start_byte..end_byte];

            // create snippet
            let mut snippet = Snippet::source(content)
                .origin(&file_info.debug_info_path)
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

        for (footer_kind, footer_message) in &footers {
            let level_mapped = match footer_kind {
                FooterKind::Info => Level::Info,
                FooterKind::Hint => Level::Help,
            };
            message = message.footer(level_mapped.title(footer_message));
        }

        if let Some(backtrace) = &backtrace
            && settings.backtrace
        {
            message = message.footer(Level::Info.title(backtrace));
        }

        // format into string
        let renderer = if settings.ansi_color {
            let color = match level {
                DiagnosticLevel::Error => AnsiColor::BrightRed,
                DiagnosticLevel::Warning => AnsiColor::BrightYellow,
            };
            Renderer::styled().emphasis(Style::new().bold().fg_color(Some(Color::Ansi(color))))
        } else {
            Renderer::plain()
        };

        let render = renderer.render(message);
        render.to_string()
    }
}

pub fn diags_to_string(source: &SourceDatabase, diags: Vec<Diagnostic>, ansi_color: bool) -> String {
    let settings = DiagnosticStringSettings::default(ansi_color);

    let mut s = String::new();
    for diag in diags {
        s.push_str(&diag.to_string(source, settings));
        s.push('\n');
        s.push('\n');
    }
    s
}

pub fn diags_to_string_vec(source: &SourceDatabase, diags: Vec<Diagnostic>, ansi_color: bool) -> Vec<String> {
    let settings = DiagnosticStringSettings::default(ansi_color);

    let mut s = vec![];
    for diag in diags {
        s.push(diag.to_string(source, settings));
    }
    s
}
