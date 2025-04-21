use crate::DiagnosticException;
use hwl_language::util::Never;
use hwl_language::{
    front::diagnostic::{DiagnosticStringSettings, Diagnostics, ErrorGuaranteed},
    syntax::source::SourceDatabase as RustSourceDatabase,
};
use itertools::Itertools;
use pyo3::prelude::*;

pub fn check_diags(source: &RustSourceDatabase, diags: &Diagnostics) -> Result<(), PyErr> {
    if diags.len() == 0 {
        Ok(())
    } else {
        let diags = diags.clone().finish();
        let settings = DiagnosticStringSettings::default();
        let msg = diags.into_iter().map(|d| d.to_string(source, settings)).join("\n");
        Err(DiagnosticException::new_err(msg))
    }
}

pub fn map_diag_error<T>(
    source: &RustSourceDatabase,
    diags: &Diagnostics,
    value: Result<T, ErrorGuaranteed>,
) -> Result<T, PyErr> {
    check_diags(source, diags)?;
    unwrap_diag_result(value)
}

pub fn unwrap_diag_result<T>(result: Result<T, ErrorGuaranteed>) -> Result<T, PyErr> {
    result.map_err(|_| DiagnosticException::new_err("diagnostic already reported previously"))
}

pub fn convert_diag_error(source: &RustSourceDatabase, diags: &Diagnostics, err: ErrorGuaranteed) -> PyErr {
    match map_diag_error::<Never>(source, diags, Err(err)) {
        Ok(never) => never.unreachable(),
        Err(e) => e,
    }
}
