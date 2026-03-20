use crate::server::vfs::{VfsError, VfsResult};
use fluent_uri::HostData;
use fluent_uri::enc::EStr;
use lsp_types::{FileSystemWatcher, GlobPattern, OneOf, RelativePattern, Uri};
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

/// Build a watcher that watches all files with the given name in the given base URI and all subdirectories.
/// `name` cannot contain any special glob characters.
pub fn build_watcher_any_file_with_name(base_uri: Uri, name: &str) -> FileSystemWatcher {
    // TODO check for features, pick relative or non-relative
    // TODO check for the presence of certain features with a proper error message
    // https://github.com/rust-lang/rust-analyzer/blob/58e507d80728f6f32c93117668dc4510ba80bac9/crates/rust-analyzer/src/reload.rs#L562-L603
    FileSystemWatcher {
        glob_pattern: GlobPattern::Relative(RelativePattern {
            base_uri: OneOf::Right(base_uri),
            pattern: format!("**/{name}"),
        }),
        // watch all file changes
        kind: None,
    }
}

// TODO maybe we want to support non-path files too, as temporary files that haven't been saved yet?
//  or should those just get their own virtual file system each?
pub fn uri_to_path(uri: &Uri) -> Result<PathBuf, VfsError> {
    // check that the URI is just a path
    // TODO check that this works on Windows, Linux and with different LSP clients
    let auth_ok = uri.authority().is_some_and(|a| {
        a.userinfo().is_none() && a.host().data() == HostData::RegName(EStr::new("")) && a.port().is_none()
    });
    let uri_ok = uri.scheme().map(|s| s.as_str()) == Some("file")
        && auth_ok
        && uri.query().is_none()
        && uri.fragment().is_none();

    if !uri_ok {
        return Err(VfsError::InvalidPathUri(uri.clone()));
    }
    // TODO always do decoding or only for some LSP clients? does the protocol really not specify this?
    let path = uri.path().as_estr().decode().into_string().unwrap();

    // TODO this is sketchy
    let path = if cfg!(windows) {
        match path.strip_prefix('/') {
            Some(path) => path,
            None => return Err(VfsError::InvalidPathUri(uri.clone())),
        }
    } else {
        &*path
    };

    Ok(PathBuf::from(path))
}

// TODO steal all of this from rust-analyzer
// TODO remove this?
pub fn abs_path_to_uri(path: &Path) -> VfsResult<Uri> {
    if !path.is_absolute() {
        return Err(VfsError::ExpectedAbsolutePath(path.to_owned()));
    }

    let path_str = path.to_str().unwrap();
    let uri_str = if cfg!(windows) {
        format!("file:///{path_str}").replace('\\', "/")
    } else {
        format!("file://{path_str}")
    };

    Uri::from_str(&uri_str).map_err(|e| VfsError::FailedToConvertPathToUri(path.to_owned(), uri_str, e))
}

pub fn uri_join(uri: &Uri, steps: impl AsRef<Path>) -> VfsResult<Uri> {
    abs_path_to_uri(&uri_to_path(uri)?.join(steps.as_ref()))
}

#[derive(Debug)]
pub struct NormalizeError {
    pub base: PathBuf,
    pub rel: PathBuf,
}

/// Utility function to join two paths with some normalization.
///
/// Only `.` and `..` components originating from the `rel` path are normalized,
/// the `base` path is preserved as much as possible.
///
/// This is useful in LSPs because we don't want to fully normalize paths,
/// that risks divergence between client and server.
///
/// The implementation is partially based on the unstable stdlib function `Path.normalize_lexically`, specially this
/// version:
/// https://github.com/rust-lang/rust/blob/fd0c901b00ee1e08a250039cdb90258603497e20/library/std/src/path.rs#L3365
pub fn path_join_normalized(base: &Path, rel: &Path) -> Result<PathBuf, NormalizeError> {
    let mut result = base.to_owned();

    let mut prev_was_prefix = false;

    for step in rel.components() {
        match step {
            Component::Prefix(c) => {
                result.clear();
                result.push(Component::Prefix(c));
            }
            Component::RootDir => {
                if !prev_was_prefix {
                    // on windows C:\ turns into (prefix, rootdir), and we don't want to clear twice
                    result.clear();
                }
                result.push(Component::RootDir);
            }
            Component::CurDir => {
                // skip
            }
            Component::ParentDir => {
                // pop last component
                if !result.pop() {
                    return Err(NormalizeError {
                        base: base.to_owned(),
                        rel: rel.to_owned(),
                    });
                }
            }
            Component::Normal(c) => {
                result.push(Component::Normal(c));
            }
        }

        prev_was_prefix = matches!(step, Component::Prefix(_));
    }

    Ok(result)
}
