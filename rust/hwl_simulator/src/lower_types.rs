use crate::simulator::{LowerError, LowerResult};
use hwl_language::mid::ir::{IrPort, IrSignals, IrType, IrWire};
use hwl_language::syntax::pos::Span;
use hwl_language::util::data::IndexMapExt;
use indexmap::IndexMap;
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::types::{ArrayType, BasicTypeEnum, StructType};

#[derive(Debug)]
pub struct ModuleSignalTypes<'ctx> {
    pub port_indices: IndexMap<IrPort, usize>,
    pub wire_indices: IndexMap<IrWire, usize>,

    /// Array of (untyped) pointers, one per port.
    /// Used for module instances, where each port is a pointer to the right signal in the parent.
    pub ports_array_ty: ArrayType<'ctx>,
    /// Struct type, one field per port.
    /// Used to store top-level ports.
    pub ports_struct_ty: StructType<'ctx>,
    /// Struct type, one field per wire.
    /// Used to store module instance wires.
    pub wires_struct_ty: StructType<'ctx>,
}

impl<'ctx> ModuleSignalTypes<'ctx> {
    pub fn new(ctx: &'ctx Context, span: Span, signals: &IrSignals) -> Result<ModuleSignalTypes<'ctx>, LowerError> {
        let IrSignals {
            ports,
            wires,
            ports_named: _,
        } = signals;

        // map ports
        let mut port_indices = IndexMap::new();
        let mut port_types = vec![];
        for (port, port_info) in ports {
            port_indices.insert_first(port, port_indices.len());

            let ty = lower_ty(ctx, &port_info.ty);
            port_types.push(ty);
        }
        let ports_struct_ty = ctx.struct_type(&port_types, false);
        let ports_array_ty = ctx
            .ptr_type(AddressSpace::default())
            .array_type(usize_to_u31(span, port_indices.len())?);

        // map wires
        let mut wire_indices = IndexMap::new();
        let mut wire_types = vec![];
        for (wire, wire_info) in wires {
            wire_indices.insert_first(wire, wire_indices.len());

            let ty = lower_ty(ctx, &wire_info.ty);
            wire_types.push(ty);
        }
        let wires_struct_ty = ctx.struct_type(&wire_types, false);

        Ok(ModuleSignalTypes {
            port_indices,
            wire_indices,
            ports_array_ty,
            ports_struct_ty,
            wires_struct_ty,
        })
    }
}

pub fn lower_ty<'ctx>(ctx: &'ctx Context, ty: &IrType) -> BasicTypeEnum<'ctx> {
    // TODO cache these? maybe that's even necessary for struct types
    // TODO optimizations:
    //   for structs and tuples, re-order to minimize size?
    //   for all compound types: bit pack? careful about multithreading!
    match ty {
        IrType::Bool => BasicTypeEnum::IntType(ctx.bool_type()),
        IrType::Int(_) => todo!(),
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}

pub fn usize_to_u31(span: Span, value: usize) -> LowerResult<u32> {
    if value < i32::MAX as usize {
        Ok(value as u32)
    } else {
        Err(LowerError::IntTooLarge(span, value.into()))
    }
}
