use check::{check_diags, convert_diag_error, unwrap_diag_error};
use convert::{compile_value_to_py, convert_python_args};
use hwl_language::{
    front::{
        compile::{
            CompileState, CompileStateLong, ModuleElaborationInfo, NoPrintHandler, PopulatedScopes, Port as RustPort,
        },
        context::CompileTimeExpressionContext,
        diagnostic::Diagnostics,
        function::FunctionValue,
        ir::IrModule,
        lower_verilog::lower,
        misc::ScopedEntry,
        scope::Visibility,
        types::{IncRange as RustIncRange, Type as RustType},
        value::MaybeCompile,
    },
    syntax::{
        ast::Spanned,
        parsed::{AstRefModule, ParsedDatabase as RustParsedDatabase},
        source::{SourceDatabase as RustSourceDatabase, SourceSetError, SourceSetOrIoError},
    },
    util::{io::IoErrorWithPath, ResultExt},
};
use itertools::{enumerate, Itertools};
use num_bigint::BigInt;
use pyo3::{
    create_exception,
    exceptions::PyException,
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::path::Path;

mod check;
mod convert;

#[pyclass]
struct Source {
    source: RustSourceDatabase,
}

#[pyclass]
struct Parsed {
    source: Py<Source>,
    parsed: RustParsedDatabase,
}

#[pyclass]
struct Compile {
    parsed: Py<Parsed>,
    state: CompileStateLong,
}

#[pyclass]
struct Undefined;

#[pyclass]
struct Type(RustType);

#[pyclass]
struct IncRange {
    #[pyo3(get)]
    start_inc: Option<BigInt>,
    #[pyo3(get)]
    end_inc: Option<BigInt>,
}

#[pyclass]
struct Function {
    compile: Py<Compile>,
    function_value: FunctionValue,
}

#[pyclass]
struct Module {
    compile: Py<Compile>,
    module: AstRefModule,
}

#[pyclass]
struct ModuleInstance {
    #[allow(dead_code)]
    compile: Py<Compile>,
    #[allow(dead_code)]
    ir_module: IrModule,
    #[allow(dead_code)]
    ports: Vec<RustPort>,
}

#[pyclass]
struct ModuleVerilog {
    #[pyo3(get)]
    module_name: String,
    #[pyo3(get)]
    source: String,
}

#[pyclass]
struct Simulator {}

create_exception!(hwl, HwlException, PyException);
create_exception!(hwl, SourceSetException, HwlException);
create_exception!(hwl, DiagnosticException, HwlException);
create_exception!(hwl, ResolveException, HwlException);
create_exception!(hwl, ValueException, HwlException);

#[pymodule]
fn hwl(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Source>()?;
    m.add_class::<Parsed>()?;
    m.add_class::<Compile>()?;
    m.add_class::<Undefined>()?;
    m.add_class::<Type>()?;
    m.add_class::<IncRange>()?;
    m.add_class::<Function>()?;
    m.add_class::<Module>()?;
    m.add_class::<ModuleInstance>()?;
    m.add_class::<ModuleVerilog>()?;
    m.add_class::<Simulator>()?;
    m.add("HwlException", py.get_type::<HwlException>())?;
    m.add("SourceSetException", py.get_type::<SourceSetException>())?;
    m.add("DiagnosticException", py.get_type::<DiagnosticException>())?;
    m.add("ResolveException", py.get_type::<ResolveException>())?;
    m.add("ValueException", py.get_type::<ValueException>())?;
    Ok(())
}

#[pymethods]
impl Source {
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

    fn parse(slf: Py<Self>, py: Python) -> PyResult<Parsed> {
        let diags = Diagnostics::new();
        let source = slf.borrow(py);
        let parsed = RustParsedDatabase::new(&diags, &source.source);

        check_diags(&source.source, &diags)?;

        drop(source);

        Ok(Parsed { source: slf, parsed })
    }
}

#[pymethods]
impl Parsed {
    fn compile(slf: Py<Self>, py: Python) -> PyResult<Compile> {
        let state = {
            let diags = Diagnostics::new();
            let parsed = slf.borrow(py);
            let source = parsed.source.borrow(py);

            let scopes = PopulatedScopes::new(&diags, &source.source, &parsed.parsed);
            let state = CompileStateLong::new(scopes.scopes, scopes.file_scopes);

            check_diags(&source.source, &diags)?;
            state
        };

        Ok(Compile { parsed: slf, state })
    }
}

#[pymethods]
impl Compile {
    fn resolve(slf: Py<Self>, py: Python, path: &str) -> PyResult<Py<PyAny>> {
        let value = {
            // unwrap self
            let slf_ref = &mut *slf.borrow_mut(py);
            let state = &mut slf_ref.state;

            let parsed_ref = slf_ref.parsed.borrow(py);
            let parsed = &parsed_ref.parsed;
            let source = &parsed_ref.source.borrow(py).source;

            // find directory, file and scope
            if path.is_empty() {
                return Err(ResolveException::new_err("resolve path cannot be empty"));
            }
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
            let scope = state.file_scopes.get(&file).unwrap().as_ref_ok().unwrap();

            // look up the item
            let diags = Diagnostics::new();
            let found = unwrap_diag_error(
                source,
                &diags,
                state.scopes[scope.scope_outer_declare].find_immediate_str(&diags, item_name, Visibility::Public),
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
            let mut print_handler = NoPrintHandler;
            let mut state_short = CompileState::new(&diags, source, parsed, state, &mut print_handler);

            let value = state_short.eval_item(item);
            let value = unwrap_diag_error(source, &diags, value)?;
            value.clone()
        };

        compile_value_to_py(py, &slf, value)
    }
}

#[pymethods]
impl Type {
    fn __str__(&self) -> String {
        self.0.to_diagnostic_string()
    }
}

#[pymethods]
impl IncRange {
    #[new]
    fn new(start_inc: Option<BigInt>, end_inc: Option<BigInt>) -> Self {
        Self { start_inc, end_inc }
    }

    fn __str__(&self) -> String {
        let range = RustIncRange {
            start_inc: self.start_inc.as_ref(),
            end_inc: self.end_inc.as_ref(),
        };
        format!("IncRange({range})")
    }
}

#[pymethods]
impl Function {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let diags = Diagnostics::new();

        let returned = {
            // borrow self
            let compile_ref = &mut *self.compile.borrow_mut(py);
            let state = &mut compile_ref.state;
            let parsed_ref = compile_ref.parsed.borrow(py);
            let parsed = &parsed_ref.parsed;
            let source = &parsed_ref.source.borrow(py).source;
            let dummy_span = parsed[self.function_value.item].common_info().span_short;

            // convert args
            let f_arg = |v| Spanned::new(dummy_span, MaybeCompile::Compile(v));
            let args = convert_python_args(args, kwargs, dummy_span, f_arg)?;

            // call function
            let mut print_handler = NoPrintHandler;
            let mut state_short = CompileState::new(&diags, source, parsed, state, &mut print_handler);
            check_diags(source, &diags)?;

            let mut ctx = CompileTimeExpressionContext {
                span: dummy_span,
                reason: "external call".to_owned(),
            };
            let returned = self.function_value.call(&mut state_short, &mut ctx, args);
            let ((), returned) = unwrap_diag_error(source, &diags, returned)?;

            // unwrap compile
            match returned {
                MaybeCompile::Compile(returned) => returned,
                MaybeCompile::Other(_) => {
                    let err = diags.report_internal_error(
                        dummy_span,
                        "function called with only compile-time args should return compile-time value",
                    );
                    return Err(convert_diag_error(source, &diags, err));
                }
            }
        };

        compile_value_to_py(py, &self.compile, returned)
    }
}

#[pymethods]
impl Module {
    #[pyo3(signature = (*args, **kwargs))]
    fn instance(
        &self,
        py: Python,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<ModuleInstance> {
        // borrow self
        let compile_ref = &mut *self.compile.borrow_mut(py);
        let state = &mut compile_ref.state;
        let parsed_ref = compile_ref.parsed.borrow(py);
        let parsed = &parsed_ref.parsed;
        let source = &parsed_ref.source.borrow(py).source;
        let dummy_span = parsed[self.module].id.span;

        // evaluate args
        let f_arg = |v| Spanned::new(dummy_span, v);
        let args = convert_python_args(args, kwargs, dummy_span, f_arg)?;
        // TODO this is really hacky, maybe the Some/None disctintion should not exist for generics?
        let args = if args.inner.len() == 0 && parsed[self.module].params.is_none() {
            None
        } else {
            Some(args)
        };

        // instantiate module
        let diags = Diagnostics::new();
        let mut print_handler = NoPrintHandler;

        let mut state_short = CompileState::new(&diags, source, parsed, state, &mut print_handler);

        let elaboration_info = ModuleElaborationInfo {
            item: self.module,
            args,
        };
        let (ir_module, ports) = unwrap_diag_error(source, &diags, state_short.elaborate_module(elaboration_info))?;

        let module_instance = ModuleInstance {
            compile: self.compile.clone_ref(py),
            ir_module,
            ports,
        };
        Ok(module_instance)
    }
}

#[pymethods]
impl ModuleInstance {
    fn to_verilog(&self, py: Python) -> PyResult<ModuleVerilog> {
        // borrow self
        let compile = self.compile.borrow(py);
        let parsed_red = compile.parsed.borrow(py);
        let parsed = &parsed_red.parsed;
        let source = &parsed_red.source.borrow(py).source;

        // lower
        let diags = Diagnostics::new();
        let lowered = lower(&diags, source, parsed, &compile.state.ir_modules, self.ir_module);
        let lowerd = unwrap_diag_error(source, &diags, lowered)?;
        Ok(ModuleVerilog {
            module_name: lowerd.top_module_name,
            source: lowerd.verilog_source,
        })
    }
}

#[pymethods]
impl Simulator {}
