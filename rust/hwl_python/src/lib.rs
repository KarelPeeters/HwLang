use std::{path::Path, vec};

use hwl_language::{
    syntax::source::{SourceDatabase, SourceSetError, SourceSetOrIoError},
    util::io::IoErrorWithPath,
};
use pyo3::{create_exception, exceptions::PyException, prelude::*};

#[pyclass]
struct Sources {
    db: SourceDatabase,
}

// declare custom exception
create_exception!(hwl, SourceSetException, PyException);

#[pymethods]
impl Sources {
    #[new]
    fn new(root_dir: &str) -> PyResult<Self> {
        let mut db = SourceDatabase::new();
        db.add_tree(vec![], Path::new(root_dir)).map_err(|e| match e {
            SourceSetOrIoError::SourceSet(source_set_error) => match source_set_error {
                SourceSetError::EmptyPath => SourceSetException::new_err("empty path"),
                SourceSetError::DuplicatePath(file_path) => {
                    SourceSetException::new_err(format!("duplicate path `{file_path:?}`"))
                }
                SourceSetError::NonUtf8Path(path_buf) => {
                    SourceSetException::new_err(format!("non-UTF-8 path `{path_buf:?}`"))
                }
                SourceSetError::MissingFileName(path_buf) => {
                    SourceSetException::new_err(format!("missing file name `{path_buf:?}`"))
                }
            },
            SourceSetOrIoError::Io(IoErrorWithPath { error, path }) => {
                SourceSetException::new_err(format!("io error `{error}` for path `{path:?}`"))
            }
        })?;
        Ok(Self { db })
    }

    #[getter]
    fn files(&self) -> Vec<String> {
        self.db
            .files()
            .into_iter()
            .map(|id| self.db[id].path_raw.clone())
            .collect()
    }
}

#[pyclass]
pub struct Simulator {}

#[pymethods]
impl Simulator {}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn hwl(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sources>()?;
    m.add_class::<Simulator>()?;
    m.add("SourceSetException", py.get_type::<SourceSetException>())?;
    Ok(())
}
