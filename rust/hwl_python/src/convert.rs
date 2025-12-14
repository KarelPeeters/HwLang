use crate::{Compile, Function, Module, Range, Type, Value};
use hwl_language::front::value::{CompileCompoundValue, SimpleCompileValue};
use hwl_language::syntax::pos::Spanned;
use hwl_language::util::big_int::BigInt;
use hwl_language::util::data::{EmptyVec, GrowVec, NonEmptyVec};
use hwl_language::util::range::Range as RustRange;
use hwl_language::{
    front::{types::Type as RustType, value::CompileValue},
    syntax::{
        ast::{Arg, Args},
        pos::Span,
    },
};
use itertools::Itertools;
use pyo3::types::{PyBool, PyInt, PyType};
use pyo3::{
    IntoPyObjectExt,
    exceptions::PyException,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use std::sync::Arc;

pub fn compile_value_to_py(py: Python, state: &Py<Compile>, value: &CompileValue) -> PyResult<Py<PyAny>> {
    // TODO get rid of this, actually expose all values to python
    let fallback = || {
        Value {
            compile: state.clone_ref(py),
            value: value.clone(),
        }
        .into_py_any(py)
    };

    match value {
        CompileValue::Simple(value) => match value {
            SimpleCompileValue::Type(x) => Type {
                compile: state.clone_ref(py),
                ty: x.clone(),
            }
            .into_py_any(py),
            SimpleCompileValue::Bool(x) => x.into_py_any(py),
            SimpleCompileValue::Int(x) => x.clone().into_num_bigint().into_py_any(py),
            SimpleCompileValue::Array(x) => {
                let items: Vec<_> = x
                    .iter()
                    .map(|item| compile_value_to_py(py, state, item))
                    .try_collect()?;
                items.into_py_any(py)
            }
            &SimpleCompileValue::Module(m) => {
                let m = Module {
                    compile: state.clone_ref(py),
                    module: m,
                };
                m.into_py_any(py)
            }
            SimpleCompileValue::Function(f) => {
                let f = Function {
                    compile: state.clone_ref(py),
                    function_value: f.clone(),
                };
                f.into_py_any(py)
            }
            SimpleCompileValue::Interface(_) => fallback(),
            SimpleCompileValue::InterfaceView(_) => fallback(),
        },
        CompileValue::Compound(value) => match value {
            CompileCompoundValue::String(x) => x.as_str().into_py_any(py),
            CompileCompoundValue::Range(x) => Range { range: x.clone() }.into_py_any(py),
            CompileCompoundValue::Tuple(x) => {
                let items: Vec<_> = x
                    .iter()
                    .map(|item| compile_value_to_py(py, state, item))
                    .try_collect()?;
                PyTuple::new(py, items.into_iter())?.into_py_any(py)
            }
            CompileCompoundValue::Struct(_) => fallback(),
            CompileCompoundValue::Enum(_) => fallback(),
        },
        CompileValue::Hardware(n) => n.unreachable(),
    }
}

pub fn compile_value_from_py(value: &Bound<PyAny>) -> PyResult<CompileValue> {
    let py = value.py();

    // TODO should we use downcast or extract here?
    //   https://pyo3.rs/v0.22.3/performance#extract-versus-downcast
    if let Ok(value) = value.extract::<PyRef<Value>>() {
        return Ok(value.value.clone());
    }
    if let Ok(value) = value.extract::<PyRef<Type>>() {
        return Ok(CompileValue::new_ty(value.ty.clone()));
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(CompileValue::new_bool(value));
    }
    if let Ok(value) = value.extract::<num_bigint::BigInt>() {
        return Ok(CompileValue::new_int(BigInt::from_num_bigint(value)));
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(CompileValue::Compound(CompileCompoundValue::String(Arc::new(value))));
    }
    if let Ok(value) = value.downcast::<PyTuple>() {
        let items: Vec<_> = value.iter().map(|v| compile_value_from_py(&v)).try_collect()?;
        return match NonEmptyVec::try_from(items) {
            Ok(items) => Ok(CompileValue::Compound(CompileCompoundValue::Tuple(items))),
            Err(EmptyVec) => Ok(CompileValue::unit()),
        };
    }
    if let Ok(value) = value.downcast::<PyList>() {
        let items: Vec<_> = value.into_iter().map(|v| compile_value_from_py(&v)).try_collect()?;
        return Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(items))));
    }
    if let Ok(value) = value.extract::<PyRef<Range>>() {
        return Ok(CompileValue::Compound(CompileCompoundValue::Range(value.range.clone())));
    }
    if let Ok(module) = value.extract::<PyRef<Module>>() {
        return Ok(CompileValue::Simple(SimpleCompileValue::Module(module.module)));
    }
    if let Ok(value) = value.extract::<PyRef<Function>>() {
        // TODO avoid clone?
        return Ok(CompileValue::Simple(SimpleCompileValue::Function(
            value.function_value.clone(),
        )));
    }

    // convert some python types with obvious equivalents
    if let Ok(py_type) = value.downcast::<PyType>() {
        if py_type.is(&py.get_type::<PyBool>()) {
            return Ok(CompileValue::new_ty(RustType::Bool));
        }
        if py_type.is(&py.get_type::<PyInt>()) {
            return Ok(CompileValue::new_ty(RustType::Int(RustRange::OPEN)));
        }
        if py_type.is(&py.get_type::<PyAny>()) {
            return Ok(CompileValue::new_ty(RustType::Any));
        }
    }

    Err(PyException::new_err(format!(
        "cannot convert value of type `{}` to compile-time value",
        value.get_type()
    )))
}

pub fn convert_python_args_and_kwargs_to_args<'k>(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    dummy_span: Span,
    key_buffer: &'k GrowVec<String>,
) -> PyResult<Args<Option<Spanned<&'k str>>, Spanned<CompileValue>>> {
    let mut args_inner = vec![];
    for value in args {
        args_inner.push(Arg {
            span: dummy_span,
            name: None,
            value: Spanned::new(dummy_span, compile_value_from_py(&value)?),
        });
    }
    if let Some(kwargs) = kwargs {
        for (name, value) in kwargs {
            let name = key_buffer.push(name.extract::<String>()?);
            args_inner.push(Arg {
                span: dummy_span,
                name: Some(Spanned::new(dummy_span, name.as_str())),
                value: Spanned::new(dummy_span, compile_value_from_py(&value)?),
            });
        }
    }

    Ok(Args {
        span: dummy_span,
        inner: args_inner,
    })
}
