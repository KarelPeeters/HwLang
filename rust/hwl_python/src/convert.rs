use crate::{Compile, Function, IncRange, Module, Type, Undefined, UnsupportedValue};
use hwl_language::util::big_int::BigInt;
use hwl_language::{
    front::{types::IncRange as RustIncRange, value::CompileValue},
    syntax::{
        ast::{Arg, Args, Identifier},
        pos::Span,
    },
};
use itertools::Itertools;
use pyo3::{
    exceptions::PyException,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
    IntoPyObjectExt,
};

pub fn compile_value_to_py(py: Python, state: &Py<Compile>, value: CompileValue) -> PyResult<Py<PyAny>> {
    match value {
        CompileValue::Undefined => Undefined.into_py_any(py),
        CompileValue::Type(x) => Type(x).into_py_any(py),
        CompileValue::Bool(x) => x.into_py_any(py),
        CompileValue::Int(x) => x.into_num_bigint().into_py_any(py),
        CompileValue::String(x) => x.into_py_any(py),
        CompileValue::Tuple(x) => {
            let items: Vec<_> = x
                .into_iter()
                .map(|item| compile_value_to_py(py, state, item))
                .try_collect()?;
            PyTuple::new(py, items.into_iter())?.into_py_any(py)
        }
        CompileValue::Array(x) => {
            let items: Vec<_> = x
                .into_iter()
                .map(|item| compile_value_to_py(py, state, item))
                .try_collect()?;
            items.into_py_any(py)
        }
        CompileValue::IntRange(x) => {
            let RustIncRange { start_inc, end_inc } = x;
            IncRange {
                start_inc: start_inc.map(BigInt::into_num_bigint).clone(),
                end_inc: end_inc.map(BigInt::into_num_bigint).clone(),
            }
            .into_py_any(py)
        }
        CompileValue::Module(m) => {
            let m = Module {
                compile: state.clone_ref(py),
                module: m,
            };
            m.into_py_any(py)
        }
        CompileValue::Function(f) => {
            let f = Function {
                compile: state.clone_ref(py),
                function_value: f,
            };
            f.into_py_any(py)
        }
        // TODO actually expose these to python
        CompileValue::Interface(_) => UnsupportedValue("interface".to_owned()).into_py_any(py),
        CompileValue::InterfaceView(_) => UnsupportedValue("interface view".to_owned()).into_py_any(py),
        CompileValue::Struct(_, _, _) => UnsupportedValue("struct".to_owned()).into_py_any(py),
        CompileValue::Enum(_, _, _) => UnsupportedValue("enum".to_owned()).into_py_any(py),
    }
}

pub fn compile_value_from_py(value: Bound<PyAny>) -> PyResult<CompileValue> {
    // TODO should we use downcast or extract here?
    //   https://pyo3.rs/v0.22.3/performance#extract-versus-downcast
    // TODO convert some obvious python types: int, bool, range, types.any
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
        return Ok(CompileValue::String(value));
    }
    if let Ok(value) = value.downcast::<PyTuple>() {
        let items: Vec<_> = value.iter().map(compile_value_from_py).try_collect()?;
        return Ok(CompileValue::Tuple(items));
    }
    if let Ok(value) = value.downcast::<PyList>() {
        let items: Vec<_> = value.into_iter().map(compile_value_from_py).try_collect()?;
        return Ok(CompileValue::Array(items));
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

    Err(PyException::new_err(format!(
        "cannot convert value of type `{}` to compile-time value",
        value.get_type()
    )))
}

pub fn convert_python_args<T>(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    dummy_span: Span,
    f: impl Fn(CompileValue) -> T,
) -> PyResult<Args<Option<Identifier>, T>> {
    let mut args_inner = vec![];
    for value in args {
        let value = compile_value_from_py(value)?;
        args_inner.push(Arg {
            span: dummy_span,
            name: None,
            value: f(value),
        });
    }
    if let Some(kwargs) = kwargs {
        for (name, value) in kwargs {
            let name = name.extract::<String>()?;
            let value = compile_value_from_py(value)?;
            let name = Some(Identifier {
                span: dummy_span,
                string: name,
            });
            args_inner.push(Arg {
                span: dummy_span,
                name,
                value: f(value),
            });
        }
    }

    Ok(Args {
        span: dummy_span,
        inner: args_inner,
    })
}
