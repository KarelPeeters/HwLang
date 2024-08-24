use crate::data::source::SourceDatabase;
use crate::error::DiagnosticError;
use crate::syntax::ast::Identifier;
use crate::syntax::pos::{DifferentFile, Span};
use annotate_snippets::renderer::{AnsiColor, Color, Style};
use annotate_snippets::{Level, Renderer, Snippet};
use std::cmp::min;
// TODO move to more common module

// TODO double-check that this was actually finished in the drop implementation? same for snippet
// TODO switch to different error collection system to support multiple errors and warnings
#[must_use]
pub struct Diagnostic<'d> {
    title: String,
    snippets: Vec<(Span, Vec<Annotation>)>,
    footers: Vec<(Level, String)>,

    // This is only stored here to make the finish call slightly neater,
    //   but could be removed again if the lifetimes are too tricky.
    database: &'d SourceDatabase,
}

#[must_use]
pub struct DiagnosticSnippet<'d> {
    diag: Diagnostic<'d>,
    span: Span,
    annotations: Vec<Annotation>,
}

struct Annotation {
    level: Level,
    span: Span,
    label: String,
}

/// The number of additional lines to show before and after each snippet range.
const SNIPPET_CONTEXT_LINES: usize = 2;
/// The maximum distance between two snippets to merge them into one.
/// This distance is measured after the context lines have already been added.
/// If `None`, no merging is done.
const SNIPPET_MERGE_MAX_DISTANCE: Option<usize> = Some(3);

impl<'d> Diagnostic<'d> {
    pub fn new(database: &'d SourceDatabase, title: impl Into<String>) -> Self {
        Diagnostic {
            title: title.into(),
            snippets: vec![],
            footers: vec![],
            database,
        }
    }

    pub fn snippet(self, span: Span) -> DiagnosticSnippet<'d> {
        DiagnosticSnippet {
            diag: self,
            span,
            annotations: vec![],
        }
    }

    pub fn footer(mut self, level: Level, footer: impl Into<String>) -> Self {
        self.footers.push((level, footer.into()));
        self
    }

    pub fn finish(self) -> DiagnosticError {
        let Self { title, snippets, footers, database } = self;
        assert!(!snippets.is_empty(), "Diagnostic without any snippets is not allowed");

        // combine snippets that are close together
        let snippets_merged = if let Some(snippet_merge_max_distance) = SNIPPET_MERGE_MAX_DISTANCE {
            // TODO fix O(n^2) complexity
            let mut snippets_merged: Vec<(Span, Vec<Annotation>)> = vec![];

            for (span, mut annotations) in snippets {
                // try merging with previous snippet
                let mut merged = false;
                for (span_prev, ref mut annotations_prev) in &mut snippets_merged {
                    // calculate distance
                    let span_full = self.database.expand_span(span);
                    let span_prev_full = self.database.expand_span(*span_prev);
                    let distance = span_full
                        .distance_lines(span_prev_full);

                    // check distance
                    let merge = match distance {
                        Ok(distance) =>
                            distance <= 2 * SNIPPET_CONTEXT_LINES + snippet_merge_max_distance,
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
            let start_line_0 = span_snippet.start.line_0.saturating_sub(SNIPPET_CONTEXT_LINES);
            let end_line_0 = min(span_snippet.end.line_0 + SNIPPET_CONTEXT_LINES, offsets.line_count() - 1);
            let start_byte = offsets.line_start_byte(start_line_0);
            let end_byte = offsets.line_start_byte(end_line_0);
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
        let string = renderer.render(message).to_string();
        DiagnosticError { string }
    }
}

impl DiagnosticAddable for Diagnostic<'_> {
    fn add(self, level: Level, span: Span, label: impl Into<String>) -> Self {
        self.snippet(span).add(level, span, label).finish()
    }
}

impl<'d> DiagnosticSnippet<'d> {
    pub fn finish(self) -> Diagnostic<'d> {
        let Self { mut diag, span, annotations } = self;
        assert!(!annotations.is_empty(), "DiagnosticSnippet without any annotations is not allowed");
        diag.snippets.push((span, annotations));
        diag
    }
}

impl DiagnosticAddable for DiagnosticSnippet<'_> {
    fn add(mut self, level: Level, span: Span, label: impl Into<String>) -> Self {
        assert!(self.span.contains(span), "DiagnosticSnippet labels must fall within snippet span");
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

pub trait DiagnosticContext {
    fn diagnostic(&self, title: impl Into<String>) -> Diagnostic<'_>;

    fn diagnostic_defined_twice(&self, kind: &str, span: Span, prev: &Identifier, curr: &Identifier) -> DiagnosticError {
        self.diagnostic(format!("duplicate {:?}", kind))
            .snippet(span)
            .add_info(prev.span, "previously defined here")
            .add_error(curr.span, "defined for the second time here")
            .finish()
            .finish()
    }

    fn diagnostic_simple(&self, title: impl Into<String>, span: Span, label: impl Into<String>) -> DiagnosticError {
        self.diagnostic(title)
            .add_error(span, label)
            .finish()
    }

    #[track_caller]
    fn diagnostic_todo(&self, span: Span, feature: &str) -> ! {
        let message = format!("feature not yet implemented: '{}'", feature);
        let err = self.diagnostic(&message)

            .add_error(span, "used here")
            .finish();
        eprintln!("{}", err.string);
        panic!("{}", message)
    }
}

impl DiagnosticContext for SourceDatabase {
    fn diagnostic(&self, title: impl Into<String>) -> Diagnostic<'_> {
        Diagnostic::new(self, title)
    }
}
