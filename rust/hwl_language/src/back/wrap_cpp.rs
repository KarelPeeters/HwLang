use crate::back::lower_cpp_wrap::{CppSignalInfo, collect_cpp_signals};
use crate::front::check::{TypeContainsReason, check_type_contains_value};
use crate::front::diagnostic::{DiagError, Diagnostics};
use crate::front::item::ElaborationArenas;
use crate::front::value::CompileValue;
use crate::mid::bits::{FromBitsInvalidValue, FromBitsWrongLength, ToBitsWrongType};
use crate::mid::ir::{IrModule, IrModules, IrPort, IrPortInfo};
use crate::syntax::ast::PortDirection;
use crate::syntax::pos::Spanned;
use crate::util::arena::{Arena, IndexType};
use crate::util::big_int::BigUint;
use dlopen2::wrapper::{Container, WrapperApi};
use indexmap::IndexMap;
use itertools::{Either, Itertools, enumerate};
use num_integer::div_ceil;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub enum CppSimError {
    Library(dlopen2::Error),
    CheckFailed,
    ApiError(u8, &'static str),
    PortTooLarge(BigUint),
    InternalError(&'static str),
    FromBitsInvalidValue(Vec<bool>, String),
    SetOutputPort(String),
}

impl Display for CppSimError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CppSimError::Library(err) => write!(f, "Library error: {err}"),
            CppSimError::CheckFailed => write!(f, "Check failed, the library and IR modules don't match"),
            CppSimError::ApiError(code, context) => write!(f, "API error {code} in context '{context}'"),
            CppSimError::PortTooLarge(size) => write!(f, "Port or signal size too large: {size} bits"),
            CppSimError::InternalError(msg) => write!(f, "Internal error: {msg}"),
            CppSimError::FromBitsInvalidValue(bits, ty) => write!(f, "Invalid bit pattern for type {ty}: {bits:?}"),
            CppSimError::SetOutputPort(port) => write!(f, "Cannot set output port `{port}`"),
        }
    }
}

impl From<dlopen2::Error> for CppSimError {
    fn from(err: dlopen2::Error) -> Self {
        CppSimError::Library(err)
    }
}

#[derive(WrapperApi)]
struct CppApi {
    check_hash: unsafe extern "C" fn() -> u64,
    create_instance: unsafe extern "C" fn() -> *mut c_void,
    destroy_instance: unsafe extern "C" fn(instance: *mut c_void),
    step: unsafe extern "C" fn(instance: *mut c_void, increment_time: u64) -> u8,
    get_port: unsafe extern "C" fn(instance: *mut c_void, port_index: usize, data_len: usize, data: *mut u8) -> u8,
    set_port: unsafe extern "C" fn(instance: *mut c_void, port_index: usize, data_len: usize, data: *const u8) -> u8,
    get_signal: unsafe extern "C" fn(instance: *mut c_void, signal_index: usize, data_len: usize, data: *mut u8) -> u8,
}

#[derive(Clone)]
pub struct CppSimLib {
    lib: Arc<Container<CppApi>>,
    ports: Arena<IrPort, IrPortInfo>,
    ports_named: IndexMap<String, IrPort>,
    signals: Arc<Vec<CppSignalInfo>>,
}

pub struct CppSimInstance {
    lib: CppSimLib,
    instance: *mut c_void,
}

#[must_use]
pub enum SimulationFinished {
    No,
    Yes,
}

impl CppSimLib {
    /// # Safety
    /// The loaded library at `path` must have been generated from the same `modules`
    /// and `top_module` as the `check_hash` passed here.
    pub unsafe fn new(
        modules: &IrModules,
        top_module: IrModule,
        check_hash: u64,
        path: &Path,
    ) -> Result<Self, CppSimError> {
        let lib = unsafe {
            let lib: Container<CppApi> = Container::load(path)?;
            let check_hash_actual = lib.check_hash();
            if check_hash_actual != check_hash {
                return Err(CppSimError::CheckFailed);
            }
            lib
        };

        let top_info = &modules[top_module];
        let ports_named = top_info.ports.iter().map(|(p, info)| (info.name.clone(), p)).collect();
        Ok(Self {
            lib: Arc::new(lib),
            ports: top_info.ports.clone(),
            ports_named,
            signals: Arc::new(collect_cpp_signals(modules, top_module)),
        })
    }

    pub fn instance(&self) -> Result<CppSimInstance, CppSimError> {
        let instance = unsafe { self.lib.create_instance() };
        if instance.is_null() {
            return Err(CppSimError::InternalError("create_instance returned null"));
        }
        Ok(CppSimInstance {
            lib: self.clone(),
            instance,
        })
    }

    pub fn ports(&self) -> &Arena<IrPort, IrPortInfo> {
        &self.ports
    }

    pub fn ports_named(&self) -> &IndexMap<String, IrPort> {
        &self.ports_named
    }

    pub fn signals(&self) -> &[CppSignalInfo] {
        &self.signals
    }
}

impl Drop for CppSimInstance {
    fn drop(&mut self) {
        unsafe {
            self.lib.lib.destroy_instance(self.instance);
        }
    }
}

impl CppSimInstance {
    pub fn step(&mut self, increment_time: u64) -> Result<SimulationFinished, CppSimError> {
        unsafe {
            let step_result = self.lib.lib.step(self.instance, increment_time);
            check_result(step_result, "step")?;
            Ok(SimulationFinished::No)
        }
    }

    pub fn ports(&self) -> &Arena<IrPort, IrPortInfo> {
        self.lib.ports()
    }

    pub fn ports_named(&self) -> &IndexMap<String, IrPort> {
        self.lib.ports_named()
    }

    pub fn signals(&self) -> &[CppSignalInfo] {
        self.lib.signals()
    }

    pub fn get_port(&self, port: IrPort) -> Result<CompileValue, CppSimError> {
        let port_info = &self.lib.ports()[port];
        let bits = self.read_port_bits(port.inner().index(), &port_info.ty)?;
        port_info.ty.value_from_bits(&bits).map_err(|e| match e {
            Either::Left(FromBitsInvalidValue) => {
                CppSimError::FromBitsInvalidValue(bits, format!("{:?}", port_info.ty))
            }
            Either::Right(FromBitsWrongLength) => CppSimError::InternalError("from bits wrong length"),
        })
    }

    pub fn set_port(
        &mut self,
        diags: &Diagnostics,
        elab: &ElaborationArenas,
        port: IrPort,
        value: Spanned<&CompileValue>,
    ) -> Result<(), Either<CppSimError, DiagError>> {
        let port_info = &self.lib.ports()[port];
        match port_info.direction {
            PortDirection::Input => {}
            PortDirection::Output => return Err(Either::Left(CppSimError::SetOutputPort(port_info.name.clone()))),
        }

        let reason = TypeContainsReason::Assignment {
            span_target: port_info.debug_span,
            span_target_ty: port_info.debug_info_ty.span,
        };
        check_type_contains_value(diags, elab, reason, &port_info.ty.as_type_hw().as_type(), value)
            .map_err(Either::Right)?;

        let bits = port_info
            .ty
            .value_to_bits(value.inner)
            .map_err(|_: ToBitsWrongType| Either::Left(CppSimError::InternalError("to_bits failed")))?;

        let size_bits = bits.len();
        let size_bytes = port_size_bytes(size_bits);
        let mut buffer = vec![0u8; size_bytes];
        pack_bits(&bits, &mut buffer);

        if size_bits != 0 {
            unsafe {
                let result = self.lib.lib.set_port(self.instance, port.inner().index(), size_bytes, buffer.as_ptr());
                check_result(result, "set_port").map_err(Either::Left)?;
            }
        }

        Ok(())
    }

    pub fn get_signal_bits(&self, signal_id: usize) -> Result<Vec<bool>, CppSimError> {
        let signal = self
            .signals()
            .get(signal_id)
            .ok_or(CppSimError::ApiError(1, "get_signal"))?;
        self.read_signal_bits(signal_id, &signal.ty)
    }

    pub fn get_signal(&self, signal_id: usize) -> Result<CompileValue, CppSimError> {
        let signal = self
            .signals()
            .get(signal_id)
            .ok_or(CppSimError::ApiError(1, "get_signal"))?;
        let bits = self.read_signal_bits(signal_id, &signal.ty)?;
        signal.ty.value_from_bits(&bits).map_err(|e| match e {
            Either::Left(FromBitsInvalidValue) => CppSimError::FromBitsInvalidValue(bits, format!("{:?}", signal.ty)),
            Either::Right(FromBitsWrongLength) => CppSimError::InternalError("from bits wrong length"),
        })
    }

    fn read_port_bits(&self, port_index: usize, ty: &crate::mid::ir::IrType) -> Result<Vec<bool>, CppSimError> {
        let size_bits = usize::try_from(ty.size_bits()).map_err(CppSimError::PortTooLarge)?;
        let size_bytes = port_size_bytes(size_bits);
        let mut buffer = vec![0u8; size_bytes];
        if size_bits != 0 {
            unsafe {
                let result = self.lib.lib.get_port(self.instance, port_index, size_bytes, buffer.as_mut_ptr());
                check_result(result, "get_port")?;
            }
        }
        Ok(unpack_bits(&buffer, size_bits))
    }

    fn read_signal_bits(&self, signal_id: usize, ty: &crate::mid::ir::IrType) -> Result<Vec<bool>, CppSimError> {
        let size_bits = usize::try_from(ty.size_bits()).map_err(CppSimError::PortTooLarge)?;
        let size_bytes = port_size_bytes(size_bits);
        let mut buffer = vec![0u8; size_bytes];
        if size_bits != 0 {
            unsafe {
                let result = self
                    .lib
                    .lib
                    .get_signal(self.instance, signal_id, size_bytes, buffer.as_mut_ptr());
                check_result(result, "get_signal")?;
            }
        }
        Ok(unpack_bits(&buffer, size_bits))
    }
}

pub(crate) fn port_size_bytes(size_bits: usize) -> usize {
    match size_bits {
        0 => 0,
        1..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        33..=64 => 8,
        65.. => div_ceil(size_bits, 32) * 4,
    }
}

fn pack_bits(bits: &[bool], buffer: &mut [u8]) {
    for (i, bit) in enumerate(bits) {
        if *bit {
            buffer[i / 8] |= 1 << (i % 8);
        }
    }
}

fn unpack_bits(buffer: &[u8], size_bits: usize) -> Vec<bool> {
    (0..size_bits)
        .map(|i| (buffer[i / 8] >> (i % 8)) & 1 != 0)
        .collect_vec()
}

fn check_result(result: u8, context: &'static str) -> Result<(), CppSimError> {
    if result != 0 {
        Err(CppSimError::ApiError(result, context))
    } else {
        Ok(())
    }
}
