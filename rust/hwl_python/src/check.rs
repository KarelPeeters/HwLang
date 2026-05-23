use crate::{Diagnostic, DiagnosticException, DiagnosticPreviouslyReportedException};
use hwl_language::front::diagnostic::{
    DiagError, DiagResult, DiagnosticLevel, Diagnostics, diag_to_string, diags_to_string,
};
use hwl_language::syntax::source::SourceDatabase as RustSourceDatabase;
use hwl_language::util::Never;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub fn check_diags(py: Python, source: &RustSourceDatabase, diags: &Diagnostics) -> Result<(), PyErr> {
    if diags.len() == 0 {
        Ok(())
    } else {
        let diags = diags.clone().finish();

        let py_diagnostics = diags.iter().map(|diag| {
            let level = match diag.level {
                DiagnosticLevel::Error => "error",
                DiagnosticLevel::Warning => "warning",
            };
            Diagnostic {
                level: level.to_owned(),
                title: diag.content.title.clone(),
                messages: diag.content.messages.iter().map(|(_, s)| s.clone()).collect_vec(),
                infos: diag.content.infos.iter().map(|(_, s)| s.clone()).collect_vec(),
                full_string: diag_to_string(source, diag, false),
                full_string_colored: diag_to_string(source, diag, true),
            }
        });

        let exc = DiagnosticException {
            diagnostics: PyList::new(py, py_diagnostics)?.unbind(),
            combined_string: diags_to_string(source, &diags, false),
            combined_string_colored: diags_to_string(source, &diags, true),
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
    unwrap_diag_result(value)
}

pub fn unwrap_diag_result<T>(result: DiagResult<T>) -> Result<T, PyErr> {
    match result {
        Ok(result) => Ok(result),
        Err(e) => {
            let _: DiagError = e;
            let msg = "diagnostic already reported previously";
            Err(DiagnosticPreviouslyReportedException::new_err(msg))
        }
    }
}

pub fn convert_diag_error(py: Python, diags: &Diagnostics, source: &RustSourceDatabase, err: DiagError) -> PyErr {
    match map_diag_error::<Never>(py, diags, source, Err(err)) {
        Ok(never) => never.unreachable(),
        Err(e) => e,
    }
}
