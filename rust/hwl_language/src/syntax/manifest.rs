use crate::syntax::source::{SourceDatabaseBuilder, SourceSetOrIoError};
use indexmap::IndexMap;
use std::path::{Path, PathBuf};

// TODO maybe add default elaboration root(s)
// TODO allow manifests to refer to other manifests
// TODO add external verilog filelist, at some point we can parse it to avoid manual "external" declarations
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub source: ManifestSourceNode,
}

impl Manifest {
    pub fn from_toml(src: &str) -> Result<Manifest, toml::de::Error> {
        // Note: it's important that the `toml` crate feature `preserve_order` is enabled,
        //   since the order of fields in the manifest can be significant, especially for the resulting error messages.
        toml::from_str(src)
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum ManifestSourceNode {
    Leaf(PathBuf),
    Branch(IndexMap<String, ManifestSourceNode>),
}

// TODO add stdlib by default
//   maybe include stdlib in the compiler itself, but still allow for an stdlib environment override
//   for stdlib development
pub fn manifest_collect_sources(
    manifest_parent: &Path,
    builder: &mut SourceDatabaseBuilder,
    prefix: &mut Vec<String>,
    node: ManifestSourceNode,
) -> Result<(), SourceSetOrIoError> {
    match node {
        ManifestSourceNode::Leaf(path) => builder.add_tree(prefix.clone(), &manifest_parent.join(path)),
        ManifestSourceNode::Branch(map) => {
            for (k, v) in map {
                let actual_key = k != "_";
                if actual_key {
                    prefix.push(k);
                }
                manifest_collect_sources(manifest_parent, builder, prefix, v)?;
                if actual_key {
                    prefix.pop();
                }
            }
            Ok(())
        }
    }
}
