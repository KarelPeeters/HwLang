use crate::front::bits::WrongType;
use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::value::CompileValue;
use crate::mid::ir::{IrModule, IrModules, IrPort, IrPortInfo};
use crate::swriteln;
use crate::syntax::ast::{PortDirection, Spanned};
use crate::util::arena::{Arena, IndexType};
use crate::util::big_int::BigUint;
use crate::util::data::IndexMapExt;
use dlopen2::wrapper::{Container, WrapperApi};
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Either, Itertools};
use regex::{Captures, Regex};
use std::convert::identity;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU16;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub struct LoweredVerilator {
    pub source: String,
    pub top_class_name: String,
}

// TODO initialize ports to undefined or at least some valid value
pub fn lower_verilator(modules: &IrModules, top_module: IrModule) -> LoweredVerilator {
    let top_module_info = &modules[top_module];

    const TEMPLATE: &str = include_str!("verilator_template.cpp");
    let mut replacements = IndexMap::new();

    for dir in [PortDirection::Input, PortDirection::Output] {
        let prefix = match dir {
            PortDirection::Input => "set",
            PortDirection::Output => "get",
        };

        let mut f = String::new();
        for (port_index, (_, port_info)) in enumerate(&top_module_info.ports) {
            let include = match (port_info.direction, dir) {
                // input ports support set/get
                (PortDirection::Input, _) => true,
                // outputs ports only support get
                (PortDirection::Output, PortDirection::Output) => true,
                (PortDirection::Output, PortDirection::Input) => false,
            };
            if !include {
                continue;
            }

            let port_name = &port_info.name;
            swriteln!(
                f,
                "            case {port_index}: return {prefix}_port_impl(wrapper->top.{port_name}, data_len, data);"
            );
        }
        replacements.insert_first(format!("PORTS-{}", prefix.to_uppercase()), f);
    }

    let arena_random: NonZeroU16 = modules.check().inner();
    replacements.insert_first("ARENA-RANDOM".to_owned(), arena_random.to_string());
    replacements.insert_first("TOP-MODULE-INDEX".to_owned(), top_module.inner().index().to_string());

    const TOP_CLASS_NAME: &str = "VTop";
    replacements.insert_first("TOP-CLASS-NAME".to_owned(), TOP_CLASS_NAME.to_owned());

    let source = template_replace(TEMPLATE, &replacements).unwrap();
    LoweredVerilator {
        source,
        top_class_name: TOP_CLASS_NAME.to_owned(),
    }
}

#[derive(Debug)]
pub enum VerilatorError {
    Library(dlopen2::Error),
    CheckFailed,
    ApiError(u8, &'static str),
    PortTooLarge(BigUint),
    InternalError(&'static str),
    SetOutputPort(String),
}

impl Display for VerilatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VerilatorError::Library(err) => write!(f, "Library error: {err}"),
            VerilatorError::CheckFailed => write!(f, "Check failed, the library and IR modules don't match"),
            VerilatorError::ApiError(code, context) => write!(f, "API error {code} in context '{context}'"),
            VerilatorError::PortTooLarge(size) => write!(f, "Port size too large: {size} bits"),
            VerilatorError::InternalError(msg) => write!(f, "Internal error: {msg}"),
            VerilatorError::SetOutputPort(port) => write!(f, "Cannot set output port `{port}`"),
        }
    }
}

impl From<dlopen2::Error> for VerilatorError {
    fn from(err: dlopen2::Error) -> Self {
        VerilatorError::Library(err)
    }
}

#[derive(WrapperApi)]
struct VerilatedApi {
    check: unsafe extern "C" fn(arena_random: u16, top_module_index: usize) -> u8,
    create_instance: unsafe extern "C" fn() -> *mut c_void,
    destroy_instance: unsafe extern "C" fn(instance: *mut c_void),
    step: unsafe extern "C" fn(instance: *mut c_void, increment_time: u64) -> u8,
    get_port: unsafe extern "C" fn(instance: *mut c_void, port_index: usize, data_len: usize, data: *mut u8) -> u8,
    set_port: unsafe extern "C" fn(instance: *mut c_void, port_index: usize, data_len: usize, data: *const u8) -> u8,
}

// TODO rename
#[derive(Clone)]
pub struct VerilatedLib {
    lib: Arc<Container<VerilatedApi>>,

    ports: Arena<IrPort, IrPortInfo>,
    ports_named: IndexMap<String, IrPort>,
}

/// This struct is intentionally not [Send], see https://verilator.org/guide/latest/verilating.html#multithreading.
pub struct VerilatedInstance {
    lib: VerilatedLib,
    instance: *mut c_void,
}

// TODO think about and document safety
impl VerilatedLib {
    /// The loaded library at `path` must be a dynamic library compiled from:
    /// * the verilated verilog generated by [crate::back::lower_verilog::lower_to_verilog]
    /// * the wrapper C++ generated by [lower_verilator]
    ///
    /// both generated from the same `modules` and `top_module`.
    ///
    /// # Safety
    /// The conditions above must be met, there is no runtime way to fully check for them.
    /// We only make a best-effort to check that the right functions are present in the library
    /// and that the same `modules` and `top_module` were used.
    pub unsafe fn new(modules: &IrModules, top_module: IrModule, path: &Path) -> Result<Self, VerilatorError> {
        let lib: Container<VerilatedApi> = Container::load(path)?;

        let result = lib.check(modules.check().inner().get(), top_module.inner().index());
        if result != 0 {
            return Err(VerilatorError::CheckFailed);
        }

        let top_info = &modules[top_module];
        let ports_named = top_info.ports.iter().map(|(p, info)| (info.name.clone(), p)).collect();
        Ok(Self {
            lib: Arc::new(lib),
            ports: top_info.ports.clone(),
            ports_named,
        })
    }

    pub fn instance(&self) -> VerilatedInstance {
        let instance = unsafe { self.lib.create_instance() };
        VerilatedInstance {
            lib: self.clone(),
            instance,
        }
    }

    pub fn ports(&self) -> &Arena<IrPort, IrPortInfo> {
        &self.ports
    }

    pub fn ports_named(&self) -> &IndexMap<String, IrPort> {
        &self.ports_named
    }
}

impl Drop for VerilatedInstance {
    fn drop(&mut self) {
        let VerilatedInstance { lib, instance } = self;
        unsafe {
            lib.lib.destroy_instance(*instance);
        }
    }
}

// TODO make this a general trait that can be implemented for different simulation backends
impl VerilatedInstance {
    pub fn step(&mut self, increment_time: u64) -> Result<(), VerilatorError> {
        // TODO error if any port has never been set?
        unsafe { check_result(self.lib.lib.step(self.instance, increment_time), "step") }
    }

    pub fn ports(&self) -> &Arena<IrPort, IrPortInfo> {
        self.lib.ports()
    }

    pub fn ports_named(&self) -> &IndexMap<String, IrPort> {
        self.lib.ports_named()
    }

    pub fn get_port(&self, port: IrPort) -> Result<CompileValue, VerilatorError> {
        let port_info = &self.lib.ports()[port];

        let size_bits = usize::try_from(port_info.ty.size_bits()).map_err(VerilatorError::PortTooLarge)?;
        let size_bytes = size_bits.div_ceil(8);
        let mut buffer = vec![0u8; size_bytes];

        unsafe {
            let result = self
                .lib
                .lib
                .get_port(self.instance, port.inner().index(), size_bytes, buffer.as_mut_ptr());
            check_result(result, "get_port")?;
        }

        let bits = (0..size_bits)
            .map(|i| (buffer[i / 8] >> (i % 8)) & 1 != 0)
            .collect_vec();
        let value = port_info
            .ty
            .value_from_bits(&bits)
            .map_err(|_: WrongType| VerilatorError::InternalError("from_bits failed"))?;

        Ok(value)
    }

    pub fn set_port(
        &mut self,
        diags: &Diagnostics,
        port: IrPort,
        value: Spanned<&CompileValue>,
    ) -> Result<(), Either<VerilatorError, ErrorGuaranteed>> {
        let port_info = &self.lib.ports()[port];
        match port_info.direction {
            PortDirection::Input => {}
            PortDirection::Output => return Err(Either::Left(VerilatorError::SetOutputPort(port_info.name.clone()))),
        }

        let reason = TypeContainsReason::Assignment {
            span_target: port_info.debug_span,
            span_target_ty: port_info.debug_info_ty.span,
        };
        check_type_contains_compile_value(diags, reason, &port_info.ty.as_type_hw().as_type(), value, true)
            .map_err(Either::Right)?;

        let bits = port_info
            .ty
            .value_to_bits(value.inner)
            .map_err(|_: WrongType| Either::Left(VerilatorError::InternalError("to_bits failed")))?;

        let size_bytes = bits.len().div_ceil(8);
        let mut buffer = vec![0u8; size_bytes];
        for (i, bit) in enumerate(bits) {
            if bit {
                buffer[i / 8] |= 1 << (i % 8);
            }
        }

        unsafe {
            let result = self
                .lib
                .lib
                .set_port(self.instance, port.inner().index(), size_bytes, buffer.as_ptr());
            check_result(result, "set_port").map_err(Either::Left)?;
        }

        Ok(())
    }
}

fn check_result(result: u8, context: &'static str) -> Result<(), VerilatorError> {
    if result != 0 {
        Err(VerilatorError::ApiError(result, context))
    } else {
        Ok(())
    }
}

fn template_replace(template: &str, replacements: &IndexMap<String, String>) -> Result<String, String> {
    let regex = Regex::new("/\\*\\[TEMPLATE-(.+)]\\*/").unwrap();

    let mut not_found = IndexSet::new();
    let mut used = vec![false; replacements.len()];

    let result = regex.replace_all(template, |caps: &Captures| {
        assert_eq!(caps.len(), 2);
        let key = caps.get(1).unwrap().as_str();
        match replacements.get_index_of(key) {
            Some(index) => {
                used[index] = true;
                replacements[index].as_str()
            }
            None => {
                not_found.insert(key.to_owned());
                ""
            }
        }
    });

    let any_not_used = !used.iter().copied().all(identity);
    if !not_found.is_empty() || any_not_used {
        let not_used = replacements
            .keys()
            .enumerate()
            .filter_map(|(i, k)| (!used[i]).then_some(k))
            .collect_vec();
        Err(format!(
            "Template substitution failed: not found: {not_found:?}, not used: {not_used:?}"
        ))
    } else {
        Ok(result.into_owned())
    }
}
