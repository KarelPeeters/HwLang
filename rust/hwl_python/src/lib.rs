use crate::convert::compile_value_from_py;
use check::{check_diags, convert_diag_error, map_diag_error};
use convert::{compile_value_to_py, convert_python_args_and_kwargs_to_args};
use hwl_language::back::lower_verilator::{lower_verilator, LoweredVerilator};
use hwl_language::back::lower_verilog::{lower_to_verilog, LoweredVerilog};
use hwl_language::back::wrap_verilator::{VerilatedInstance as RustVerilatedInstance, VerilatedLib, VerilatorError};
use hwl_language::front::compile::{CompileFixed, CompileItemContext, CompileRefs, CompileShared, PartialIrDatabase};
use hwl_language::front::item::ElaboratedModule;
use hwl_language::front::print::StdoutPrintHandler;
use hwl_language::front::scope::ScopedEntry;
use hwl_language::front::variables::VariableValues;
use hwl_language::mid::ir::{IrModule, IrModuleInfo, IrPort, IrPortInfo};
use hwl_language::syntax::ast::{Arg, Args};
use hwl_language::syntax::pos::Span;
use hwl_language::syntax::source::FilePath;
use hwl_language::util::{ResultExt, NON_ZERO_USIZE_ONE};
use hwl_language::{
    front::{
        context::CompileTimeExpressionContext,
        diagnostic::Diagnostics,
        function::FunctionValue,
        types::{IncRange as RustIncRange, Type as RustType},
        value::Value,
    },
    syntax::{
        ast::Spanned,
        parsed::ParsedDatabase as RustParsedDatabase,
        source::{
            SourceDatabase as RustSourceDatabase, SourceDatabaseBuilder as RustSourceDatabaseBuilder, SourceSetError,
            SourceSetOrIoError,
        },
    },
};
use hwl_util::io::IoErrorWithPath;
use itertools::{enumerate, Either, Itertools};
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::types::PyIterator;
use pyo3::{
    create_exception,
    exceptions::PyException,
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::path::Path;
use std::process::Command;

mod check;
mod convert;

#[pyclass]
struct Source {
    source: RustSourceDatabase,
    dummy_span: Span,
}

#[pyclass]
struct Parsed {
    source: Py<Source>,
    parsed: RustParsedDatabase,
}

#[pyclass]
struct Compile {
    parsed: Py<Parsed>,
    state: CompileShared,
}

#[pyclass]
struct UnsupportedValue(String);

#[pymethods]
impl UnsupportedValue {
    fn __repr__(&self) -> String {
        format!("Unsupported({})", self.0)
    }
}

#[pyclass]
struct Undefined;

#[pyclass]
struct Type(RustType);

#[pyclass]
struct IncRange {
    #[pyo3(get)]
    start_inc: Option<num_bigint::BigInt>,
    #[pyo3(get)]
    end_inc: Option<num_bigint::BigInt>,
}

#[pyclass]
struct Function {
    compile: Py<Compile>,
    function_value: FunctionValue,
}

#[pyclass]
struct Module {
    compile: Py<Compile>,
    module: ElaboratedModule,
}

#[pyclass]
struct ModuleVerilog {
    #[pyo3(get)]
    module_name: String,
    #[pyo3(get)]
    source: String,
}

#[pyclass]
struct ModuleVerilated {
    compile: Py<Compile>,
    lib: VerilatedLib,
}

#[pyclass(unsendable)]
struct VerilatedInstance {
    module: Py<ModuleVerilated>,
    instance: RustVerilatedInstance,
}

#[pyclass(unsendable)]
struct VerilatedPorts {
    instance: Py<VerilatedInstance>,
}

#[pyclass(unsendable)]
struct VerilatedPort {
    instance: Py<VerilatedInstance>,
    port: IrPort,
}

create_exception!(hwl, HwlException, PyException);
create_exception!(hwl, SourceSetException, HwlException);
create_exception!(hwl, DiagnosticException, HwlException);
create_exception!(hwl, ResolveException, HwlException);
create_exception!(hwl, GenerateVerilogException, HwlException);
create_exception!(hwl, VerilationException, HwlException);

#[pymodule]
fn hwl(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Source>()?;
    m.add_class::<Parsed>()?;
    m.add_class::<Compile>()?;
    m.add_class::<UnsupportedValue>()?;
    m.add_class::<Undefined>()?;
    m.add_class::<Type>()?;
    m.add_class::<IncRange>()?;
    m.add_class::<Function>()?;
    m.add_class::<Module>()?;
    m.add_class::<ModuleVerilog>()?;
    m.add_class::<ModuleVerilated>()?;
    m.add_class::<VerilatedInstance>()?;
    m.add_class::<VerilatedPorts>()?;
    m.add_class::<VerilatedPort>()?;
    m.add("HwlException", py.get_type::<HwlException>())?;
    m.add("SourceSetException", py.get_type::<SourceSetException>())?;
    m.add("DiagnosticException", py.get_type::<DiagnosticException>())?;
    m.add("ResolveException", py.get_type::<ResolveException>())?;
    Ok(())
}

#[pymethods]
impl Source {
    #[new]
    fn new(root_dir: &str) -> PyResult<Self> {
        let mut source_builder = RustSourceDatabaseBuilder::new();

        let dummy_source = "// dummy file, representing the python caller";
        let dummy_file = source_builder
            .add_file(
                FilePath(vec!["python".to_owned(), root_dir.to_owned()]),
                "python.py".to_owned(),
                dummy_source.to_owned(),
            )
            .unwrap();

        source_builder
            .add_tree(vec![], Path::new(root_dir))
            .map_err(|e| match e {
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

        let (source, _, mapping) = source_builder.finish_with_mapping();

        let dummy_file = *mapping.get(&dummy_file).unwrap();
        let dummy_span = Span::new(dummy_file, 0, dummy_source.len());

        Ok(Self { source, dummy_span })
    }

    #[getter]
    fn files(&self) -> Vec<String> {
        self.source.files().map(|id| self.source[id].path_raw.clone()).collect()
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
            let fixed = CompileFixed {
                source: &source.source,
                parsed: &parsed.parsed,
            };

            let state = CompileShared::new(&diags, fixed, false, NON_ZERO_USIZE_ONE);
            check_diags(&source.source, &diags)?;

            state
        };

        Ok(Compile { parsed: slf, state })
    }
}

#[pymethods]
impl Compile {
    fn resolve(slf: Py<Self>, py: Python, path: &str) -> PyResult<Py<PyAny>> {
        // TODO move this somewhere common, the commandline will also need this
        // unwrap self
        let slf_ref = &mut *slf.borrow_mut(py);
        let state = &slf_ref.state;

        let parsed_ref = slf_ref.parsed.borrow(py);
        let parsed = &parsed_ref.parsed;
        let source = &parsed_ref.source.borrow(py).source;

        // find directory, file and scope
        if path.is_empty() {
            return Err(ResolveException::new_err("resolve path cannot be empty"));
        }
        let steps: Vec<&str> = path.split('.').collect_vec();
        let (item_name, steps) = steps.split_last().unwrap();

        let mut curr_dir = source.root_directory();
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
        let found = map_diag_error(source, &diags, scope.find_immediate_str(&diags, item_name))?;
        let item = match found.value {
            &ScopedEntry::Item(ast_ref_item) => ast_ref_item,
            ScopedEntry::Named(_) => {
                let e = diags.report_internal_error(
                    found.defining_span,
                    "file scope should only contain items, not named/value",
                );
                return Err(convert_diag_error(source, &diags, e));
            }
        };

        // evaluate the item
        let refs = CompileRefs {
            fixed: CompileFixed { source, parsed },
            shared: state,
            diags: &diags,
            print_handler: &StdoutPrintHandler,
            should_stop: &|| false,
        };
        let mut item_ctx = CompileItemContext::new_empty(refs, None);

        let value = item_ctx.eval_item(item);
        let value = map_diag_error(source, &diags, value)?;

        refs.run_elaboration_loop();
        check_diags(source, &diags)?;

        compile_value_to_py(py, &slf, value)
    }
}

#[pymethods]
impl Type {
    fn __str__(&self) -> String {
        self.0.diagnostic_string()
    }
}

#[pymethods]
impl IncRange {
    #[new]
    fn new(start_inc: Option<num_bigint::BigInt>, end_inc: Option<num_bigint::BigInt>) -> Self {
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
    // TODO implement this on more/all values, not just functions and modules
    //   (eg. struct/enum constructors, int type construction, ...)
    //   just follow exactly what expression eval does, ideally share most code with it
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
            let source_ref = parsed_ref.source.borrow(py);
            let source = &source_ref.source;
            let dummy_span = source_ref.dummy_span;

            // convert args
            let args = convert_python_args_and_kwargs_to_args(args, kwargs, dummy_span)?;
            let args = Args {
                span: dummy_span,
                inner: args
                    .inner
                    .iter()
                    .map(|arg| {
                        Arg {
                            span: dummy_span,
                            name: arg.name.as_ref().map(|name| Spanned::new(dummy_span, name.as_str())),
                            // TODO avoid clone here
                            value: Spanned::new(dummy_span, Value::Compile(arg.value.clone())),
                        }
                    })
                    .collect_vec(),
            };

            // call function
            let refs = CompileRefs {
                fixed: CompileFixed { source, parsed },
                shared: state,
                diags: &diags,
                print_handler: &StdoutPrintHandler,
                should_stop: &|| false,
            };

            let mut item_ctx = CompileItemContext::new_empty(refs, None);
            let mut vars = VariableValues::new_root(&item_ctx.variables);

            let mut ctx = CompileTimeExpressionContext {
                span: dummy_span,
                reason: "external call".to_owned(),
            };

            let returned = item_ctx.call_function(
                &mut ctx,
                &mut vars,
                &RustType::Any,
                dummy_span,
                dummy_span,
                &self.function_value,
                args,
            );
            let (_block, returned) = map_diag_error(source, &diags, returned)?;

            // run any downstream elaboration
            refs.run_elaboration_loop();
            check_diags(source, &diags)?;

            // unwrap compile
            match returned {
                Value::Compile(returned) => returned,
                Value::Hardware(_) => {
                    let err = diags.report_internal_error(
                        dummy_span,
                        "function called with only compile-time args should return compile-time value",
                    );
                    return Err(convert_diag_error(source, &diags, err));
                }
            }
        };

        compile_value_to_py(py, &self.compile, &returned)
    }
}

#[pymethods]
impl Module {
    fn as_verilog(&mut self, py: Python) -> PyResult<ModuleVerilog> {
        let (_, _, lowered) = self.lower_verilog_impl(py)?;
        Ok(ModuleVerilog {
            module_name: lowered.top_module_name,
            source: lowered.source,
        })
    }

    fn as_verilated(&self, build_dir: &str, py: Python) -> PyResult<ModuleVerilated> {
        // check build_dir
        let build_dir = Path::new(build_dir);
        if !build_dir.exists() {
            return Err(PyIOError::new_err(format!(
                "build_dir `{}` does not exist",
                build_dir.display()
            )));
        }
        if !build_dir.is_dir() {
            return Err(PyIOError::new_err(format!(
                "build_dir `{}` is not a directory",
                build_dir.display()
            )));
        }

        // lower
        let (ir_database, ir_module, lowered_verilog) = self.lower_verilog_impl(py)?;
        let lowered_verilator = lower_verilator(&ir_database.ir_modules, ir_module);

        let LoweredVerilog {
            source: source_verilog,
            top_module_name,
            debug_info_module_map: _,
        } = lowered_verilog;
        let LoweredVerilator {
            source: source_cpp,
            top_class_name,
        } = lowered_verilator;

        // write to files
        // TODO only write if changed to avoid unnecessary rebuilds? or does verilator already do that for us?
        let name_verilog = "lowered.v";
        std::fs::write(build_dir.join(name_verilog), &source_verilog)?;
        let name_cpp = "lowered.cpp";
        std::fs::write(build_dir.join(name_cpp), &source_cpp)?;

        // verilate
        // TODO get everything properly incremental
        // TODO make tracing optional
        // TODO move this compilation process to somewhere else, not in the python create
        run_command(
            Command::new("verilator")
                .arg("-cc")
                .arg("-CFLAGS")
                .arg("-fPIC")
                // TODO improve backend so these are no longer needed?
                .arg("-Wno-widthexpand")
                .arg("-Wno-cmpconst")
                .arg("-Wno-widthtrunc")
                .arg("+1364-2001ext+v")
                .arg("--trace")
                .arg("--top-module")
                .arg(&top_module_name)
                .arg("--prefix")
                .arg(&top_class_name)
                .arg(name_verilog)
                .arg(name_cpp),
            build_dir,
            "verilator",
        )?;

        // compile
        let obj_dir = build_dir.join("obj_dir");
        run_command(
            Command::new("make")
                .arg("-f")
                .arg(format!("{}.mk", top_class_name))
                .arg("-j")
                .arg(num_cpus::get().to_string()),
            &obj_dir,
            "make",
        )?;

        // link
        let objects = std::fs::read_dir(&obj_dir)
            .map_err(|e| VerilationException::new_err(format!("failed to read obj_dir: {}", e)))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "o"))
            .map(|entry| entry.file_name())
            .collect::<Vec<_>>();

        let name_so = "combined.so";
        let path_so = obj_dir.join(name_so);

        // TODO use faster linker
        run_command(
            Command::new("g++").args(objects).arg("-o").arg(name_so).arg("-shared"),
            &obj_dir,
            "linking",
        )?;

        // load library
        let lib = unsafe {
            VerilatedLib::new(&ir_database.ir_modules, ir_module, &path_so)
                .map_err(|e| VerilationException::new_err(format!("lib loading failed: {e:?}")))?
        };
        Ok(ModuleVerilated {
            compile: self.compile.clone_ref(py),
            lib,
        })
    }
}

// TODO include stderr in the error message
//   https://users.rust-lang.org/t/best-error-handing-practices-when-using-std-command/42259
fn run_command(command: &mut Command, dir: &Path, name: &str) -> PyResult<()> {
    let status = command
        .current_dir(dir)
        .status()
        .map_err(|e| VerilationException::new_err(format!("`{name}` failed to launch: error={e:?}")))?;

    if !status.success() {
        return Err(VerilationException::new_err(format!(
            "`{name}` failed: code={:?}",
            status.code()
        )));
    }

    Ok(())
}

impl Module {
    fn lower_verilog_impl(&self, py: Python) -> PyResult<(PartialIrDatabase<IrModuleInfo>, IrModule, LoweredVerilog)> {
        // borrow self
        let compile = self.compile.borrow(py);
        let parsed_ref = compile.parsed.borrow(py);
        let source_ref = parsed_ref.source.borrow(py);
        let source = &source_ref.source;
        let dummy_span = source_ref.dummy_span;

        // get the module
        let module = match self.module {
            ElaboratedModule::Internal(module) => module,
            ElaboratedModule::External(_) => {
                return Err(GenerateVerilogException::new_err(
                    "cannot generate verilog for external module",
                ))
            }
        };
        let ir_module = compile.state.elaboration_arenas.module_internal_info(module).module_ir;

        // create temporary ir database
        let diags = Diagnostics::new();
        let ir_database = compile.state.finish_ir_database_ref(&diags, dummy_span);
        let ir_database = map_diag_error(source, &diags, ir_database)?;

        // actual lowering
        let lowered = lower_to_verilog(
            &diags,
            &ir_database.ir_modules,
            &ir_database.external_modules,
            ir_module,
        );
        let lowered = map_diag_error(source, &diags, lowered)?;

        Ok((ir_database, ir_module, lowered))
    }
}

#[pymethods]
impl ModuleVerilated {
    #[pyo3(signature=(trace_path=None))]
    fn instance(slf: Py<Self>, trace_path: Option<&str>, py: Python) -> PyResult<VerilatedInstance> {
        let instance = slf
            .borrow(py)
            .lib
            .instance(trace_path.map(Path::new))
            .map_err(map_verilator_error)?;
        Ok(VerilatedInstance {
            module: slf.clone_ref(py),
            instance,
        })
    }
}

#[pymethods]
impl VerilatedInstance {
    #[getter]
    fn ports(slf: Py<Self>) -> VerilatedPorts {
        VerilatedPorts { instance: slf }
    }

    fn step(&mut self, increment_time: u64) -> PyResult<()> {
        let instance = &mut self.instance;
        instance.step(increment_time).map_err(map_verilator_error)?;
        Ok(())
    }

    fn save_trace(&mut self) {
        self.instance.save_trace();
    }
}

#[pymethods]
impl VerilatedPorts {
    fn __getattr__(&self, attr: &str, py: Python) -> PyResult<VerilatedPort> {
        let port = self.get_port(attr, py)?;
        Ok(VerilatedPort {
            instance: self.instance.clone_ref(py),
            port,
        })
    }

    fn __getitem__(&self, key: &str, py: Python) -> PyResult<VerilatedPort> {
        self.__getattr__(key, py)
    }

    fn __setattr__(&mut self, attr: &str, value: Py<PyAny>, py: Python) -> PyResult<()> {
        // setting values directly is not actually allowed, but we can return a nicer error message than
        //   the misleading "ports does not have attribute"
        let _ = self.get_port(attr, py)?;
        let _ = value;

        let msg = format!("cannot set port value directly, use `ports.{attr}.value = value` instead)");
        Err(PyValueError::new_err(msg))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let ports = self
            .instance
            .borrow(py)
            .module
            .borrow(py)
            .lib
            .ports_named()
            .keys()
            .cloned()
            .collect_vec();
        ports.into_pyobject(py)?.try_iter()
    }
}

impl VerilatedPorts {
    fn get_port(&self, name: &str, py: Python) -> PyResult<IrPort> {
        let instance = &self.instance.borrow(py).instance;
        instance
            .ports_named()
            .get(name)
            .copied()
            .ok_or_else(|| PyKeyError::new_err(format!("port {name} not found")))
    }
}

// TODO think about how this should look for interface ports, we probably want a "proper" mapping somewhere
// TODO change this so .value is no longer needed, probably with separate `ports` and `info_ports` fields on the simulator
#[pymethods]
impl VerilatedPort {
    #[getter]
    fn get_value(&self, py: Python) -> PyResult<Py<PyAny>> {
        let instance = self.instance.borrow(py);
        let compile = &instance.module.borrow(py).compile;

        let value = instance.instance.get_port(self.port).map_err(map_verilator_error)?;
        compile_value_to_py(py, compile, &value)
    }

    #[setter]
    fn set_value(&mut self, value: &Bound<PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut instance = self.instance.borrow_mut(py);

        let dummy_span = {
            let module = instance.module.borrow(py);
            let compile = module.compile.borrow(py);
            let parsed = compile.parsed.borrow(py);
            let source = parsed.source.borrow(py);
            source.dummy_span
        };

        let value = compile_value_from_py(value)?;
        let diags = Diagnostics::new();
        let result = instance
            .instance
            .set_port(&diags, self.port, Spanned::new(dummy_span, &value));

        // reborrow chain, annoying but seems to be necessary
        let module = instance.module.borrow(py);
        let compile = module.compile.borrow(py);
        let parsed = compile.parsed.borrow(py);
        let source = parsed.source.borrow(py);

        result.map_err(|e| match e {
            Either::Left(e) => map_verilator_error(e),
            Either::Right(e) => convert_diag_error(&source.source, &diags, e),
        })?;

        Ok(())
    }

    #[getter]
    fn r#type(&self, py: Python) -> Type {
        self.map_port_info(py, |info| Type(info.ty.as_type_hw().as_type()))
    }

    #[getter]
    fn name(&self, py: Python) -> String {
        self.map_port_info(py, |info| info.name.clone())
    }

    #[getter]
    fn direction(&self, py: Python) -> &'static str {
        self.map_port_info(py, |info| info.direction.diagnostic_string())
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyValueError::new_err(
            "port cannot be used as a boolean, to read a boolean port use `port.value` instead",
        ))
    }
}

impl VerilatedPort {
    pub fn map_port_info<T>(&self, py: Python, f: impl FnOnce(&IrPortInfo) -> T) -> T {
        let instance = self.instance.borrow(py);
        let module = instance.module.borrow(py);
        f(&module.lib.ports()[self.port])
    }
}

fn map_verilator_error(e: VerilatorError) -> PyErr {
    VerilationException::new_err(e.to_string())
}
