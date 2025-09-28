use crate::DiagnosticException;
use hwl_language::front::diagnostic::{DiagResult, diags_to_string};
use hwl_language::util::Never;
use hwl_language::{
    front::diagnostic::{DiagError, Diagnostics},
    syntax::source::SourceDatabase as RustSourceDatabase,
};
use pyo3::prelude::*;

pub fn check_diags(source: &RustSourceDatabase, diags: &Diagnostics) -> Result<(), PyErr> {
    if diags.len() == 0 {
        Ok(())
    } else {
        // TODO should we include ansi colors in python exceptions?
        //   if it works it looks nice, but we don't know where it will be displayed
        let msg = diags_to_string(source, diags.clone().finish(), true);
        Err(DiagnosticException::new_err(msg))
    }
}

pub fn map_diag_error<T>(diags: &Diagnostics, source: &RustSourceDatabase, value: DiagResult<T>) -> Result<T, PyErr> {
    check_diags(source, diags)?;
    unwrap_diag_result(value)
}

pub fn unwrap_diag_result<T>(result: DiagResult<T>) -> Result<T, PyErr> {
    result.map_err(|_| DiagnosticException::new_err("diagnostic already reported previously"))
}

pub fn convert_diag_error(diags: &Diagnostics, source: &RustSourceDatabase, err: DiagError) -> PyErr {
    match map_diag_error::<Never>(diags, source, Err(err)) {
        Ok(never) => never.unreachable(),
        Err(e) => e,
    }
}
