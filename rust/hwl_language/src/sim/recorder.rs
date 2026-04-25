use crate::back::lower_cpp_wrap::{CppSignalInfo, CppSignalKind};
use crate::back::wrap_cpp::{CppSimError, CppSimInstance, port_size_bytes};
use crate::mid::ir::{IrEnumType, IrType};
use crate::util::big_int::BigUint;
use crate::util::int::{IntRepresentation, Signed};
use itertools::enumerate;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveStore {
    pub signals: Vec<WaveSignal>,
    pub changes: Vec<Vec<WaveChange>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveSignal {
    pub id: usize,
    pub path: Vec<String>,
    pub name: String,
    #[serde(default)]
    pub kind: WaveSignalKind,
    pub ty: WaveSignalType,
    pub bit_len: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub enum WaveSignalKind {
    Port,
    #[default]
    Wire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveChange {
    pub time: u64,
    pub bits: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WaveSignalType {
    Bool,
    Int {
        signed: bool,
        width: usize,
    },
    Array {
        len: usize,
        element: Box<WaveSignalType>,
    },
    Tuple(Vec<WaveSignalType>),
    Struct {
        name: String,
        fields: Vec<(String, WaveSignalType)>,
    },
    Enum {
        name: String,
        variants: Vec<(String, Option<WaveSignalType>)>,
    },
}

impl WaveStore {
    pub fn new(signals: &[CppSignalInfo]) -> Result<Self, CppSimError> {
        let signals = signals
            .iter()
            .map(WaveSignal::from_cpp_signal)
            .collect::<Result<Vec<_>, _>>()?;
        let changes = vec![Vec::new(); signals.len()];
        Ok(Self { signals, changes })
    }

    pub fn for_instance(instance: &CppSimInstance) -> Result<Self, CppSimError> {
        Self::new(instance.signals())
    }

    pub fn sample(&mut self, instance: &CppSimInstance, time: u64) -> Result<(), CppSimError> {
        for signal in &self.signals {
            let bits = instance.get_signal_bits(signal.id)?;
            let packed = pack_bits(&bits);
            let signal_changes = &mut self.changes[signal.id];
            let changed = signal_changes
                .last()
                .is_none_or(|last| last.bits.as_slice() != packed.as_slice());
            if changed {
                signal_changes.push(WaveChange { time, bits: packed });
            }
        }
        Ok(())
    }

    pub fn signal_value_at(&self, signal_id: usize, time: u64) -> Option<&[u8]> {
        self.changes.get(signal_id)?.iter().rev().find_map(|change| {
            if change.time <= time {
                Some(change.bits.as_slice())
            } else {
                None
            }
        })
    }

    pub fn max_time(&self) -> u64 {
        self.changes
            .iter()
            .filter_map(|signal_changes| signal_changes.last().map(|change| change.time))
            .max()
            .unwrap_or(0)
    }
}

impl WaveSignal {
    fn from_cpp_signal(signal: &CppSignalInfo) -> Result<Self, CppSimError> {
        let bit_len = usize::try_from(signal.ty.size_bits()).map_err(CppSimError::PortTooLarge)?;
        Ok(Self {
            id: signal.id,
            path: signal.path.clone(),
            name: signal.name.clone(),
            kind: WaveSignalKind::from_cpp_signal_kind(signal.kind),
            ty: WaveSignalType::from_ir_type(&signal.ty)?,
            bit_len,
        })
    }
}

impl WaveSignalKind {
    fn from_cpp_signal_kind(kind: CppSignalKind) -> Self {
        match kind {
            CppSignalKind::Port => WaveSignalKind::Port,
            CppSignalKind::Wire => WaveSignalKind::Wire,
        }
    }
}

impl WaveSignalType {
    fn from_ir_type(ty: &IrType) -> Result<Self, CppSimError> {
        match ty {
            IrType::Bool => Ok(WaveSignalType::Bool),
            IrType::Int(range) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                let width = usize::try_from(repr.size_bits())
                    .map_err(|_| CppSimError::PortTooLarge(BigUint::from(repr.size_bits())))?;
                Ok(WaveSignalType::Int {
                    signed: repr.signed() == Signed::Signed,
                    width,
                })
            }
            IrType::Array(inner, len) => Ok(WaveSignalType::Array {
                len: usize::try_from(len.clone()).map_err(CppSimError::PortTooLarge)?,
                element: Box::new(WaveSignalType::from_ir_type(inner)?),
            }),
            IrType::Tuple(elements) => Ok(WaveSignalType::Tuple(
                elements
                    .iter()
                    .map(WaveSignalType::from_ir_type)
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            IrType::Struct(info) => Ok(WaveSignalType::Struct {
                name: info.debug_info_name.clone(),
                fields: info
                    .fields
                    .iter()
                    .map(|(name, ty)| Ok((name.clone(), WaveSignalType::from_ir_type(ty)?)))
                    .collect::<Result<Vec<_>, CppSimError>>()?,
            }),
            IrType::Enum(info) => enum_type_to_wave(info),
        }
    }

    pub fn bit_len(&self) -> usize {
        match self {
            WaveSignalType::Bool => 1,
            WaveSignalType::Int { width, .. } => *width,
            WaveSignalType::Array { len, element } => len * element.bit_len(),
            WaveSignalType::Tuple(elements) => elements.iter().map(WaveSignalType::bit_len).sum(),
            WaveSignalType::Struct { fields, .. } => fields.iter().map(|(_, ty)| ty.bit_len()).sum(),
            WaveSignalType::Enum { variants, .. } => {
                let tag_width = enum_tag_width(variants.len());
                let payload_width = variants
                    .iter()
                    .filter_map(|(_, ty)| ty.as_ref().map(WaveSignalType::bit_len))
                    .max()
                    .unwrap_or(0);
                tag_width + payload_width
            }
        }
    }
}

fn enum_type_to_wave(info: &IrEnumType) -> Result<WaveSignalType, CppSimError> {
    Ok(WaveSignalType::Enum {
        name: info.debug_info_name.clone(),
        variants: info
            .variants
            .iter()
            .map(|(name, ty)| Ok((name.clone(), ty.as_ref().map(WaveSignalType::from_ir_type).transpose()?)))
            .collect::<Result<Vec<_>, CppSimError>>()?,
    })
}

fn enum_tag_width(variant_count: usize) -> usize {
    if variant_count <= 1 {
        0
    } else {
        usize::BITS as usize - (variant_count - 1).leading_zeros() as usize
    }
}

fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut buffer = vec![0u8; port_size_bytes(bits.len())];
    for (i, bit) in enumerate(bits) {
        if *bit {
            buffer[i / 8] |= 1 << (i % 8);
        }
    }
    buffer
}
