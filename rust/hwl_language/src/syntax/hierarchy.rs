use crate::front::diagnostic::{DiagError, DiagnosticError, Diagnostics};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::str_is_valid_identifier;
use indexmap::IndexMap;

#[derive(Debug)]
pub struct SourceHierarchy {
    root: HierarchyNode,
    files: IndexMap<FileId, Vec<String>>,
}

// TODO cross-platform ordering guarantees?
#[derive(Debug, Default)]
pub struct HierarchyNode {
    pub file: Option<FileId>,
    pub children: IndexMap<String, HierarchyNode>,
}

impl SourceHierarchy {
    pub fn new() -> Self {
        SourceHierarchy {
            root: HierarchyNode::default(),
            files: IndexMap::new(),
        }
    }

    pub fn root_node(&self) -> &HierarchyNode {
        &self.root
    }

    pub fn files(&self) -> impl Iterator<Item = FileId> + '_ {
        self.files.keys().copied()
    }

    pub fn file_steps(&self, file: FileId) -> Option<&[String]> {
        self.files.get(&file).map(|steps| steps.as_slice())
    }

    pub fn add_file(
        &mut self,
        diags: &Diagnostics,
        source: &SourceDatabase,
        span: Span,
        steps: &[String],
        file: FileId,
    ) -> Result<(), DiagError> {
        // check file not yet added
        if self.files.contains_key(&file) {
            return Err(diags.report_error_simple(
                format!("File `{}` already exists in hierarchy", source[file].debug_info_path),
                span,
                "file added here",
            ));
        }

        // check that steps are valid identifiers
        for step in steps {
            if !str_is_valid_identifier(step) {
                return Err(diags.report_error_simple(
                    format!("Invalid identifier `{step}` in hierarchy steps"),
                    span,
                    "file added here",
                ));
            }
        }

        // add the file to the right node
        let mut curr_node = &mut self.root;
        let mut steps_left = steps;
        loop {
            (curr_node, steps_left) = match steps_left.split_first() {
                None => {
                    match curr_node.file {
                        None => {
                            curr_node.file = Some(file);
                            break;
                        }
                        Some(prev_file) => {
                            // TODO separate spans
                            return Err(DiagnosticError::new(
                                format!("file with hierarchy steps `{}` already exists", steps.join(".")),
                                span,
                                format!("file with path `{}` added here", source[file].debug_info_path),
                            )
                            .add_info(
                                span,
                                format!("file with path `{}` added here", source[prev_file].debug_info_path),
                            )
                            .report(diags));
                        }
                    };
                }
                Some((step, steps_rest)) => {
                    let next_node = curr_node.children.entry(step.clone()).or_default();
                    (next_node, steps_rest)
                }
            }
        }

        // remember the file steps
        self.files.insert(file, steps.to_vec());

        Ok(())
    }
}
