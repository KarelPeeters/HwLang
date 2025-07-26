use crate::util::data::VecExt;
use indexmap::IndexMap;

// TODO maybe add default elaboration root(s)
// TODO allow manifests to refer to other manifests
// TODO add external verilog filelist, at some point we can parse it to avoid manual "external" declarations
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub source: ManifestSource,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields, transparent)]
pub struct ManifestSource {
    node: ManifestSourceNode,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ManifestSourceNode {
    Leaf(String),
    Branch(IndexMap<String, ManifestSourceNode>),
}

impl Manifest {
    pub fn from_toml(src: &str) -> Result<Manifest, toml::de::Error> {
        // Note: it's important that the `toml` crate feature `preserve_order` is enabled,
        //   since the order of fields in the manifest can be significant, especially for the resulting error messages.
        toml::from_str(src)
    }
}

pub struct SourceEntry {
    pub steps: Vec<String>,
    pub path_relative: String,
}

// TODO somehow add stdlib by default
impl ManifestSource {
    pub fn entries(&self) -> Vec<SourceEntry> {
        let mut entries = vec![];
        for_each_entry_impl(&mut vec![], &self.node, &mut |prefix, path| {
            let entry = SourceEntry {
                steps: prefix.iter().map(|s| s.to_string()).collect(),
                path_relative: path.to_string(),
            };
            entries.push(entry);
        });
        entries
    }
}

fn for_each_entry_impl<'n>(prefix: &mut Vec<&'n str>, node: &'n ManifestSourceNode, f: &mut impl FnMut(&[&str], &str)) {
    match node {
        ManifestSourceNode::Leaf(path) => f(prefix, path),
        ManifestSourceNode::Branch(map) => {
            for (k, v) in map {
                let actual_key = k != "_";

                if actual_key {
                    prefix.with_pushed(k, |prefix| {
                        for_each_entry_impl(prefix, v, f);
                    })
                } else {
                    for_each_entry_impl(prefix, v, f);
                }
            }
        }
    }
}
