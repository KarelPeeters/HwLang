use hwl_language::front::diagnostic::{DiagError, Diagnostics, diags_to_string};
use hwl_language::syntax::manifest::Manifest;
use hwl_language::syntax::source::{FileId, SourceDatabase};
use hwl_util::constants::HWL_MANIFEST_FILE_NAME;
use hwl_util::io::IoErrorExt;
use path_clean::PathClean;
use std::io::ErrorKind;
use std::path::PathBuf;

/// Error type indicating that all necessary errors have been printed,
/// and the program should now just exit with [ExitCode::FAILURE].
#[derive(Debug)]
pub struct ErrorExit;

#[derive(Debug)]
pub struct ParsedManifest {
    pub path: PathBuf,
    pub path_parent: PathBuf,
    pub file: FileId,
    pub parsed: Manifest,
}

pub fn manifest_find_read_parse(
    source: &mut SourceDatabase,
    manifest: Option<PathBuf>,
) -> Result<ParsedManifest, ErrorExit> {
    let found = match manifest_find_read(manifest) {
        Ok(m) => m,
        Err(FindManifestError(msg)) => {
            eprintln!("{msg}");
            return Err(ErrorExit);
        }
    };
    let manifest_file = source.add_file(found.path.to_string_lossy().into_owned(), found.source);

    let diags = Diagnostics::new();
    let manifest = match Manifest::parse_toml(&diags, source, manifest_file) {
        Ok(m) => m,
        Err(e) => {
            let _: DiagError = e;
            print_diagnostics(source, diags);
            return Err(ErrorExit);
        }
    };

    Ok(ParsedManifest {
        path: found.path,
        path_parent: found.path_parent,
        file: manifest_file,
        parsed: manifest,
    })
}

struct FindManifestError(pub String);

struct FoundManifest {
    path: PathBuf,
    path_parent: PathBuf,
    source: String,
}

fn manifest_find_read(manifest_path: Option<PathBuf>) -> Result<FoundManifest, FindManifestError> {
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
                Ok(source) => {
                    let manifest_parent = manifest_path
                        .parent()
                        .ok_or_else(|| {
                            FindManifestError(format!(
                                "Manifest path {manifest_path:?} does not have a parent directory"
                            ))
                        })?
                        .to_owned();
                    Ok(FoundManifest {
                        path: manifest_path,
                        path_parent: manifest_parent,
                        source,
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
                    Ok(source) => {
                        return Ok(FoundManifest {
                            path: cand_manifest_path,
                            path_parent: ancestor.to_owned(),
                            source,
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
