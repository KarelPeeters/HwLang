use hwl_language::front::diagnostic::{Diagnostics, diags_to_string};
use hwl_language::syntax::source::SourceDatabase;
use hwl_util::constants::HWL_MANIFEST_FILE_NAME;
use hwl_util::io::IoErrorExt;
use path_clean::PathClean;
use std::io::ErrorKind;
use std::path::PathBuf;

pub struct FoundManifest {
    pub manifest_path: PathBuf,
    pub manifest_parent: PathBuf,
    pub manifest_source: String,
}

pub struct FindManifestError(pub String);

pub fn find_and_read_manifest(manifest_path: Option<PathBuf>) -> Result<FoundManifest, FindManifestError> {
    let cwd = std::env::current_dir()
        .map_err(|e| FindManifestError(format!("Failed to get current working directory: {e:?}")))?;
    let cwd = std::path::absolute(&cwd).map_err(|e| {
        FindManifestError(format!(
            "Failed to convert working dir to absolute path: {:?}",
            e.with_path(cwd)
        ))
    })?;

    match manifest_path {
        Some(manifest_path) => {
            // directly read the manifest file
            let manifest_path = cwd.join(manifest_path).clean();
            match std::fs::read_to_string(&manifest_path) {
                Ok(s) => {
                    let manifest_parent = manifest_path
                        .parent()
                        .ok_or_else(|| {
                            FindManifestError(format!(
                                "Manifest path {manifest_path:?} does not have a parent directory"
                            ))
                        })?
                        .to_owned();
                    Ok(FoundManifest {
                        manifest_path,
                        manifest_parent,
                        manifest_source: s,
                    })
                }
                Err(e) => Err(FindManifestError(format!(
                    "Failed to read manifest file: {:?}",
                    e.with_path(manifest_path)
                ))),
            }
        }
        None => {
            // walk up the path until we find a folder containing a manifest file
            for ancestor in cwd.ancestors() {
                let cand_manifest_path = ancestor.join(HWL_MANIFEST_FILE_NAME);
                match std::fs::read_to_string(&cand_manifest_path) {
                    Ok(s) => {
                        return Ok(FoundManifest {
                            manifest_path: cand_manifest_path,
                            manifest_parent: ancestor.to_owned(),
                            manifest_source: s,
                        });
                    }
                    Err(e) => match e.kind() {
                        ErrorKind::NotFound => continue,
                        _ => {
                            return Err(FindManifestError(format!(
                                "Failed to read manifest file: {:?}",
                                e.with_path(cand_manifest_path)
                            )));
                        }
                    },
                }
            }

            Err(FindManifestError(format!(
                "No manifest file `{HWL_MANIFEST_FILE_NAME}` found in any parent directory of the current working directory {cwd:?}"
            )))
        }
    }
}

pub fn print_diagnostics(source: &SourceDatabase, diags: Diagnostics) -> bool {
    let diags = diags.finish();
    let any_error = !diags.is_empty();

    let result = diags_to_string(source, diags, true);
    eprintln!("{result}");

    any_error
}
