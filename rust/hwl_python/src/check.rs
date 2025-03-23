use hwl_language::{
    front::diagnostic::{DiagnosticStringSettings, Diagnostics, ErrorGuaranteed},
    syntax::source::SourceDatabase as RustSourceDatabase,
};
use itertools::Itertools;
use pyo3::prelude::*;

use crate::DiagnosticException;

pub fn check_diags(source: &RustSourceDatabase, diags: &Diagnostics) -> Result<(), PyErr> {
    if diags.len() == 0 {
        Ok(())
    } else {
        let diags = diags.clone_without_handler().finish();
        let settings = DiagnosticStringSettings::default();
        let msg = diags.into_iter().map(|d| d.to_string(source, settings)).join("\n");
        Err(DiagnosticException::new_err(msg))
    }
}

pub fn unwrap_diag_error<T>(
    source: &RustSourceDatabase,
    diags: &Diagnostics,
    value: Result<T, ErrorGuaranteed>,
) -> Result<T, PyErr> {
    check_diags(source, diags)?;
    Ok(value.unwrap())
}

pub fn convert_diag_error(source: &RustSourceDatabase, diags: &Diagnostics, err: ErrorGuaranteed) -> PyErr {
    let _ = err;
    check_diags(source, diags).unwrap_err()
}
