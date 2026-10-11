use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::new_index_type;
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::str_is_valid_identifier;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use itertools::enumerate;

// TODO intern strings here?
new_index_type!(pub HierarchyNode);

#[derive(Debug)]
pub struct SourceHierarchy {
    root: HierarchyNode,
    files: IndexMap<FileId, HierarchyNode>,
    nodes: Arena<HierarchyNode, HierarchyNodeInfo>,
}

#[derive(Debug)]
pub struct HierarchyNodeInfo {
    pub file: Option<FileId>,
    pub steps: Vec<String>,
    pub children: IndexMap<String, HierarchyNode>,
}

impl SourceHierarchy {
    pub fn new() -> Self {
        let mut nodes = Arena::new();

        let root_info = HierarchyNodeInfo {
            file: None,
            steps: vec![],
            children: IndexMap::new(),
        };
        let root = nodes.push(root_info);

        SourceHierarchy {
            root,
            files: IndexMap::new(),
            nodes,
        }
    }

    pub fn root_node(&self) -> HierarchyNode {
        self.root
    }

    pub fn files(&self) -> impl Iterator<Item = FileId> + '_ {
        self.files.keys().copied()
    }

    pub fn file_node(&self, file: FileId) -> Option<HierarchyNode> {
        self.files.get(&file).copied()
    }

    pub fn file_steps(&self, file: FileId) -> Option<&[String]> {
        self.file_node(file).map(|node| self.nodes[node].steps.as_slice())
    }

    pub fn add_file(
        &mut self,
        diags: &Diagnostics,
        source: &SourceDatabase,
        span: Span,
        steps: &[String],
        file: FileId,
    ) -> DiagResult<HierarchyNode> {
        // check file not yet added
        if self.files.contains_key(&file) {
            return Err(diags.report_error_simple(
                format!("File `{}` was already added as a package", source[file].debug_info_path),
                span,
                "file added here",
            ));
        }

        // check that steps are valid identifiers
        for step in steps {
            if !str_is_valid_identifier(step) {
                return Err(diags.report_error_simple(
                    format!("Invalid identifier `{step}` in package path"),
                    span,
                    "file added here",
                ));
            }
        }

        // create all necessary nodes
        let mut curr_node = self.root;
        for (step_index, step) in enumerate(steps) {
            curr_node = match self.nodes[curr_node].children.get(step) {
                Some(&child) => child,
                None => {
                    let info = HierarchyNodeInfo {
                        file: None,
                        steps: steps[..=step_index].to_vec(),
                        children: IndexMap::new(),
                    };
                    let child = self.nodes.push(info);
                    self.nodes[curr_node].children.insert_first(step.to_owned(), child);
                    child
                }
            }
        }

        // check that this is not a duplicate path
        let curr_node_ref = &mut self.nodes[curr_node];
        if let Some(prev_file) = curr_node_ref.file {
            // TODO separate spans
            return Err(DiagnosticError::new(
                format!("package `{}` already exists", steps.join(".")),
                span,
                format!("file with path `{}` added here", source[file].debug_info_path),
            )
            .add_info(
                span,
                format!("file with path `{}` added here", source[prev_file].debug_info_path),
            )
            .report(diags));
        }

        // add the file to the current node
        curr_node_ref.file = Some(file);

        // remember the file node
        self.files.insert(file, curr_node);

        Ok(curr_node)
    }
}

impl std::ops::Index<HierarchyNode> for SourceHierarchy {
    type Output = HierarchyNodeInfo;
    fn index(&self, node: HierarchyNode) -> &Self::Output {
        &self.nodes[node]
    }
}
