use itertools::{enumerate, Itertools};
use num_bigint::BigInt;
use std::{path::Path, vec};

use hwl_language::{
    front::{
        compile::{CompileState, NoPrintHandler},
        diagnostic::{DiagnosticStringSettings, Diagnostics, ErrorGuaranteed},
        misc::ScopedEntry,
        scope::Visibility,
        types::IncRange as RustIncRange,
        value::CompileValue as RustCompileValue,
    },
    syntax::{
        parsed::ParsedDatabase as RustParsedDatabase,
        source::{SourceDatabase as RustSourceDatabase, SourceSetError, SourceSetOrIoError},
    },
    util::{io::IoErrorWithPath, ResultExt},
};
use pyo3::{create_exception, exceptions::PyException, prelude::*, types::PyTuple, IntoPyObjectExt};

#[pyclass]
struct SourceDatabase {
    source: RustSourceDatabase,
}

#[pyclass]
struct ParsedDatabase {
    source: Py<SourceDatabase>,
    parsed: RustParsedDatabase,
}

#[pyclass]
struct Undefined;

#[pyclass]
struct Type(#[allow(dead_code)] String);

#[pyclass]
struct IncRange {
    #[pyo3(get)]
    start_inc: Option<BigInt>,
    #[pyo3(get)]
    end_inc: Option<BigInt>,
}

#[pyclass]
struct Module(#[allow(dead_code)] String);

#[pyclass]
struct Function;

#[pyclass]
pub struct Simulator {}

create_exception!(hwl, HwlException, PyException);
create_exception!(hwl, SourceSetException, HwlException);
create_exception!(hwl, DiagnosticException, HwlException);
create_exception!(hwl, ResolveException, HwlException);

#[pymodule]
fn hwl(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SourceDatabase>()?;
    m.add_class::<Simulator>()?;
    m.add("SourceSetException", py.get_type::<SourceSetException>())?;
    m.add("DiagnosticException", py.get_type::<DiagnosticException>())?;
    Ok(())
}

#[pymethods]
impl SourceDatabase {
    #[new]
    fn new(root_dir: &str) -> PyResult<Self> {
        let mut source = RustSourceDatabase::new();
        source.add_tree(vec![], Path::new(root_dir)).map_err(|e| match e {
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
        Ok(Self { source })
    }

    #[getter]
    fn files(&self) -> Vec<String> {
        self.source
            .files()
            .into_iter()
            .map(|id| self.source[id].path_raw.clone())
            .collect()
    }

    fn parse(slf: Py<SourceDatabase>, py: Python) -> PyResult<ParsedDatabase> {
        let diags = Diagnostics::new();
        let source = slf.borrow(py);
        let parsed = RustParsedDatabase::new(&diags, &source.source);

        check_diags(&source.source, &diags)?;

        drop(source);
        Ok(ParsedDatabase { source: slf, parsed })
    }
}

#[pymethods]
impl ParsedDatabase {
    fn resolve(&self, py: Python, path: &str) -> PyResult<Py<PyAny>> {
        if path.is_empty() {
            return Err(ResolveException::new_err("resolve path cannot be empty"));
        }

        // create temporary state
        let diags = Diagnostics::new();
        let source = &self.source.borrow(py).source;
        let mut print_handler = NoPrintHandler;
        let (mut state, _) = CompileState::new(&diags, source, &self.parsed, &mut print_handler);
        check_diags(source, &diags)?;

        // find directory, file and scope
        let steps: Vec<&str> = path.split('.').collect_vec();
        let (item_name, steps) = steps.split_last().unwrap();

        let mut curr_dir = source.root_directory;
        for (i_step, &step) in enumerate(steps) {
            curr_dir = source[curr_dir].children.get(step).copied().ok_or_else(|| {
                ResolveException::new_err(format!(
                    "path `{}` does not have child `{}`",
                    steps[..i_step].iter().join("."),
                    step
                ))
            })?;
        }
        let file = source[curr_dir].file.ok_or_else(|| {
            ResolveException::new_err(format!("path `{}` does not point to a file", steps.iter().join(".")))
        })?;

        let scope =
            unwrap_diag_error(source, &diags, state.file_scopes.get(&file).unwrap().as_ref_ok())?.scope_outer_declare;

        // look up the item
        let found = unwrap_diag_error(
            source,
            &diags,
            state.scopes[scope].find_immediate_str(&diags, item_name, Visibility::Public),
        )?;
        let item = match found.value {
            &ScopedEntry::Item(ast_ref_item) => ast_ref_item,
            ScopedEntry::Direct(_) => {
                let e = diags.report_internal_error(
                    found.defining_span,
                    "file scope should only contain items, not direct values",
                );
                return Err(convert_diag_error(source, &diags, e));
            }
        };

        // evaluate the item
        let value = state.eval_item(item);
        let value = unwrap_diag_error(source, &diags, value)?;
        compile_value_to_py(py, value.clone(), &state)
    }
}

#[pymethods]
impl Simulator {}

fn compile_value_to_py(py: Python, value: RustCompileValue, state: &CompileState) -> PyResult<Py<PyAny>> {
    match value {
        RustCompileValue::Undefined => Undefined.into_py_any(py),
        RustCompileValue::Type(x) => Type(x.to_diagnostic_string()).into_py_any(py),
        RustCompileValue::Bool(x) => x.into_py_any(py),
        RustCompileValue::Int(x) => x.into_py_any(py),
        RustCompileValue::String(x) => x.into_py_any(py),
        RustCompileValue::Tuple(x) => {
            let items: Vec<_> = x
                .into_iter()
                .map(|item| compile_value_to_py(py, item, state))
                .try_collect()?;
            PyTuple::new(py, items.into_iter())?.into_py_any(py)
        }
        RustCompileValue::Array(x) => {
            let items: Vec<_> = x
                .into_iter()
                .map(|item| compile_value_to_py(py, item, state))
                .try_collect()?;
            items.into_py_any(py)
        }
        RustCompileValue::IntRange(x) => {
            let RustIncRange { start_inc, end_inc } = x;
            IncRange {
                start_inc: start_inc.clone(),
                end_inc: end_inc.clone(),
            }
            .into_py_any(py)
        }
        RustCompileValue::Module(x) => Module(state.parsed[x].id.string.clone()).into_py_any(py),
        RustCompileValue::Function(_) => Function.into_py_any(py),
    }
}

fn check_diags(source: &RustSourceDatabase, diags: &Diagnostics) -> Result<(), PyErr> {
    if diags.len() == 0 {
        Ok(())
    } else {
        let diags = diags.clone_without_handler().finish();
        let settings = DiagnosticStringSettings::default();
        let msg = diags.into_iter().map(|d| d.to_string(source, settings)).join("\n");
        Err(DiagnosticException::new_err(msg))
    }
}

fn unwrap_diag_error<T>(
    source: &RustSourceDatabase,
    diags: &Diagnostics,
    value: Result<T, ErrorGuaranteed>,
) -> Result<T, PyErr> {
    check_diags(source, diags)?;
    Ok(value.unwrap())
}

fn convert_diag_error(source: &RustSourceDatabase, diags: &Diagnostics, err: ErrorGuaranteed) -> PyErr {
    let _ = err;
    check_diags(source, diags).unwrap_err()
}
