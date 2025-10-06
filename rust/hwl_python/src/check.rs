use crate::DiagnosticException;
use hwl_language::front::diagnostic::{DiagResult, diags_to_string_vec};
use hwl_language::util::Never;
use hwl_language::{
    front::diagnostic::{DiagError, Diagnostics},
    syntax::source::SourceDatabase as RustSourceDatabase,
};
use pyo3::prelude::*;

pub fn check_diags(py: Python, source: &RustSourceDatabase, diags: &Diagnostics) -> Result<(), PyErr> {
    if diags.len() == 0 {
        Ok(())
    } else {
        let exc = DiagnosticException {
            messages: diags_to_string_vec(source, diags.clone().finish(), false),
            messages_colored: diags_to_string_vec(source, diags.clone().finish(), true),
        };
        Err(exc.into_err(py)?)
    }
}

pub fn map_diag_error<T>(
    py: Python,
    diags: &Diagnostics,
    source: &RustSourceDatabase,
    value: DiagResult<T>,
) -> Result<T, PyErr> {
    check_diags(py, source, diags)?;
    unwrap_diag_result(py, value)
}

pub fn unwrap_diag_result<T>(py: Python, result: DiagResult<T>) -> Result<T, PyErr> {
    match result {
        Ok(result) => Ok(result),
        Err(e) => {
            let _: DiagError = e;

            // TODO create a dedicated exception type for this case?
            let msg = "diagnostic already reported previously";
            let exc = DiagnosticException {
                messages: vec![msg.to_string()],
                messages_colored: vec![msg.to_string()],
            };
            Err(exc.into_err(py)?)
        }
    }
}

pub fn convert_diag_error(py: Python, diags: &Diagnostics, source: &RustSourceDatabase, err: DiagError) -> PyErr {
    match map_diag_error::<Never>(py, diags, source, Err(err)) {
        Ok(never) => never.unreachable(),
        Err(e) => e,
    }
}
