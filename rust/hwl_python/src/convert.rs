use crate::{Compile, Function, IncRange, Module, Type, Undefined, UnsupportedValue};
use hwl_language::util::big_int::BigInt;
use hwl_language::{
    front::{types::IncRange as RustIncRange, types::Type as RustType, value::CompileValue},
    syntax::{
        ast::{Arg, Args},
        pos::Span,
    },
};
use itertools::Itertools;
use pyo3::types::{PyBool, PyInt, PyType};
use pyo3::{
    exceptions::PyException,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
    IntoPyObjectExt,
};
use std::sync::Arc;

pub fn compile_value_to_py(py: Python, state: &Py<Compile>, value: &CompileValue) -> PyResult<Py<PyAny>> {
    match value {
        CompileValue::Undefined => Undefined.into_py_any(py),
        CompileValue::Type(x) => Type(x.clone()).into_py_any(py),
        CompileValue::Bool(x) => x.into_py_any(py),
        CompileValue::Int(x) => x.clone().into_num_bigint().into_py_any(py),
        CompileValue::String(x) => x.as_str().into_py_any(py),
        CompileValue::Tuple(x) => {
            let items: Vec<_> = x
                .iter()
                .map(|item| compile_value_to_py(py, state, item))
                .try_collect()?;
            PyTuple::new(py, items.into_iter())?.into_py_any(py)
        }
        CompileValue::Array(x) => {
            let items: Vec<_> = x
                .iter()
                .map(|item| compile_value_to_py(py, state, item))
                .try_collect()?;
            items.into_py_any(py)
        }
        CompileValue::IntRange(x) => {
            let RustIncRange { start_inc, end_inc } = x;
            IncRange {
                start_inc: start_inc.clone().map(BigInt::into_num_bigint).clone(),
                end_inc: end_inc.clone().map(BigInt::into_num_bigint).clone(),
            }
            .into_py_any(py)
        }
        &CompileValue::Module(m) => {
            let m = Module {
                compile: state.clone_ref(py),
                module: m,
            };
            m.into_py_any(py)
        }
        CompileValue::Function(f) => {
            let f = Function {
                compile: state.clone_ref(py),
                function_value: f.clone(),
            };
            f.into_py_any(py)
        }
        // TODO actually expose these to python
        CompileValue::Interface(_) => UnsupportedValue("interface".to_owned()).into_py_any(py),
        CompileValue::InterfaceView(_) => UnsupportedValue("interface view".to_owned()).into_py_any(py),
        CompileValue::Struct(_, _) => UnsupportedValue("struct".to_owned()).into_py_any(py),
        CompileValue::Enum(_, _) => UnsupportedValue("enum".to_owned()).into_py_any(py),
    }
}

pub fn compile_value_from_py(value: &Bound<PyAny>) -> PyResult<CompileValue> {
    let py = value.py();

    // TODO should we use downcast or extract here?
    //   https://pyo3.rs/v0.22.3/performance#extract-versus-downcast
    if value.extract::<PyRef<Undefined>>().is_ok() {
        return Ok(CompileValue::Undefined);
    }
    if let Ok(value) = value.extract::<PyRef<Type>>() {
        return Ok(CompileValue::Type(value.0.clone()));
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(CompileValue::Bool(value));
    }
    if let Ok(value) = value.extract::<num_bigint::BigInt>() {
        return Ok(CompileValue::Int(BigInt::from_num_bigint(value)));
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(CompileValue::String(Arc::new(value)));
    }
    if let Ok(value) = value.downcast::<PyTuple>() {
        let items: Vec<_> = value.iter().map(|v| compile_value_from_py(&v)).try_collect()?;
        return Ok(CompileValue::Tuple(Arc::new(items)));
    }
    if let Ok(value) = value.downcast::<PyList>() {
        let items: Vec<_> = value.into_iter().map(|v| compile_value_from_py(&v)).try_collect()?;
        return Ok(CompileValue::Array(Arc::new(items)));
    }
    if let Ok(value) = value.extract::<PyRef<IncRange>>() {
        let IncRange { start_inc, end_inc } = &*value;
        return Ok(CompileValue::IntRange(RustIncRange {
            start_inc: start_inc.clone().map(BigInt::from_num_bigint),
            end_inc: end_inc.clone().map(BigInt::from_num_bigint),
        }));
    }
    if let Ok(module) = value.extract::<PyRef<Module>>() {
        return Ok(CompileValue::Module(module.module));
    }
    if let Ok(value) = value.extract::<PyRef<Function>>() {
        // TODO avoid clone?
        return Ok(CompileValue::Function(value.function_value.clone()));
    }

    // convert some python types with obvious equivalents
    if let Ok(py_type) = value.downcast::<PyType>() {
        if py_type.is(&py.get_type::<PyBool>()) {
            return Ok(CompileValue::Type(RustType::Bool));
        }
        if py_type.is(&py.get_type::<PyInt>()) {
            return Ok(CompileValue::Type(RustType::Int(RustIncRange::OPEN)));
        }
        if py_type.is(&py.get_type::<PyAny>()) {
            return Ok(CompileValue::Type(RustType::Any));
        }
    }

    Err(PyException::new_err(format!(
        "cannot convert value of type `{}` to compile-time value",
        value.get_type()
    )))
}

pub fn convert_python_args_and_kwargs_to_args(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    dummy_span: Span,
) -> PyResult<Args<Option<String>, CompileValue>> {
    let mut args_inner = vec![];
    for value in args {
        args_inner.push(Arg {
            span: dummy_span,
            name: None,
            value: compile_value_from_py(&value)?,
        });
    }
    if let Some(kwargs) = kwargs {
        for (name, value) in kwargs {
            let name = name.extract::<String>()?;
            args_inner.push(Arg {
                span: dummy_span,
                name: Some(name),
                value: compile_value_from_py(&value)?,
            });
        }
    }

    Ok(Args {
        span: dummy_span,
        inner: args_inner,
    })
}
