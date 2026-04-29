use crate::mid::ir::IrType;
use crate::util::Indent;
use crate::util::big_int::BigUint;
use crate::util::int::IntRepresentation;
use hwl_util::swriteln;
use itertools::enumerate;

#[derive(Debug, Copy, Clone)]
pub enum CppBitStore<'a> {
    BoolArray { array: &'a str },
    PackedData { data: &'a str },
}

#[derive(Debug, Copy, Clone)]
pub enum CppBitLoad<'a> {
    BoolArray { array: &'a str },
    PackedData { data: &'a str },
}

impl CppBitStore<'_> {
    fn write_bit(self, f: &mut String, indent: Indent, offset: &str, bit: &str) {
        match self {
            CppBitStore::BoolArray { array } => {
                swriteln!(f, "{indent}{array}[{offset}] = {bit};");
            }
            CppBitStore::PackedData { data } => {
                swriteln!(f, "{indent}hwlang_cpp_wrap::write_bit({data}, {offset}, {bit});");
            }
        }
    }
}

impl CppBitLoad<'_> {
    fn read_bit(self, offset: &str) -> String {
        match self {
            CppBitLoad::BoolArray { array } => format!("{array}[{offset}]"),
            CppBitLoad::PackedData { data } => format!("hwlang_cpp_wrap::read_bit({data}, {offset})"),
        }
    }
}

pub fn emit_value_to_bits(
    f: &mut String,
    indent: Indent,
    ty: &IrType,
    value: &str,
    offset: &str,
    store: CppBitStore,
    temp_name: &mut impl FnMut(&str, &str) -> String,
) {
    match ty {
        IrType::Bool => {
            store.write_bit(f, indent, offset, value);
        }
        IrType::Int(range) => {
            let width = IntRepresentation::for_range(range.as_ref()).size_bits();
            let tmp_i = temp_name("i", offset);
            swriteln!(
                f,
                "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {width}; {tmp_i}++) {{"
            );
            store.write_bit(
                f,
                indent.nest(),
                &format!("{offset} + {tmp_i}"),
                &format!("(({value}) >> {tmp_i}) & 1"),
            );
            swriteln!(f, "{indent}}}");
        }
        IrType::Array(inner, len) => {
            let inner_bits = inner.size_bits();
            let tmp_i = temp_name("i", offset);
            swriteln!(
                f,
                "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {len}; {tmp_i}++) {{"
            );
            emit_value_to_bits(
                f,
                indent.nest(),
                inner,
                &format!("({value})[{tmp_i}]"),
                &format!("{offset} + {tmp_i} * {inner_bits}"),
                store,
                temp_name,
            );
            swriteln!(f, "{indent}}}");
        }
        IrType::Tuple(elements) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(elements) {
                emit_value_to_bits(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                    store,
                    temp_name,
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Struct(info) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(info.fields.values()) {
                emit_value_to_bits(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                    store,
                    temp_name,
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Enum(info) => {
            emit_value_to_bits(
                f,
                indent,
                &IrType::Int(info.tag_range()),
                &format!("static_cast<int64_t>(({value}).index())"),
                offset,
                store,
                temp_name,
            );
            let payload_offset = info.tag_size_bits();
            swriteln!(f, "{indent}switch (({value}).index()) {{");
            for (variant_i, payload_ty) in enumerate(info.variants.values()) {
                if let Some(payload_ty) = payload_ty {
                    swriteln!(f, "{indent}{I}case {variant_i}: {{");
                    emit_value_to_bits(
                        f,
                        indent.nest().nest(),
                        payload_ty,
                        &format!("std::get<{variant_i}>({value})"),
                        &format!("{offset} + {payload_offset}"),
                        store,
                        temp_name,
                    );
                    swriteln!(f, "{indent}{I}{I}break;");
                    swriteln!(f, "{indent}{I}}}");
                }
            }
            swriteln!(f, "{indent}}}");
        }
    }
}

pub fn emit_value_from_bits(
    f: &mut String,
    indent: Indent,
    ty: &IrType,
    value: &str,
    offset: &str,
    load: CppBitLoad,
    temp_name: &mut impl FnMut(&str, &str) -> String,
) {
    match ty {
        IrType::Bool => {
            let bit = load.read_bit(offset);
            swriteln!(f, "{indent}{value} = {bit};");
        }
        IrType::Int(range) => {
            let repr = IntRepresentation::for_range(range.as_ref());
            let width = repr.size_bits();
            let tmp = temp_name("v", offset);
            let tmp_i = temp_name("i", offset);
            swriteln!(f, "{indent}uint64_t {tmp} = 0;");
            swriteln!(
                f,
                "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {width}; {tmp_i}++) {{"
            );
            let bit = load.read_bit(&format!("{offset} + {tmp_i}"));
            swriteln!(f, "{indent}{I}if ({bit}) {tmp} |= (uint64_t{{1}} << {tmp_i});");
            swriteln!(f, "{indent}}}");
            match repr {
                IntRepresentation::Unsigned { .. } => {
                    swriteln!(f, "{indent}{value} = static_cast<int64_t>({tmp});");
                }
                IntRepresentation::Signed { width_1 } => {
                    let sign_adjust = width_1 + 1;
                    let sign_bit = load.read_bit(&format!("{offset} + {width_1}"));
                    swriteln!(
                        f,
                        "{indent}{value} = {sign_bit} ? static_cast<int64_t>({tmp}) - (int64_t{{1}} << {sign_adjust}) : static_cast<int64_t>({tmp});"
                    );
                }
            }
        }
        IrType::Array(inner, len) => {
            let inner_bits = inner.size_bits();
            let tmp_i = temp_name("i", offset);
            swriteln!(
                f,
                "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {len}; {tmp_i}++) {{"
            );
            emit_value_from_bits(
                f,
                indent.nest(),
                inner,
                &format!("({value})[{tmp_i}]"),
                &format!("{offset} + {tmp_i} * {inner_bits}"),
                load,
                temp_name,
            );
            swriteln!(f, "{indent}}}");
        }
        IrType::Tuple(elements) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(elements) {
                emit_value_from_bits(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                    load,
                    temp_name,
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Struct(info) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(info.fields.values()) {
                emit_value_from_bits(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                    load,
                    temp_name,
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Enum(info) => {
            let tag_ty = IrType::Int(info.tag_range());
            let tag_tmp = temp_name("tag", offset);
            swriteln!(f, "{indent}int64_t {tag_tmp} = 0;");
            emit_value_from_bits(f, indent, &tag_ty, &tag_tmp, offset, load, temp_name);
            let payload_offset = info.tag_size_bits();
            swriteln!(f, "{indent}switch ({tag_tmp}) {{");
            for (variant_i, payload_ty) in enumerate(info.variants.values()) {
                swriteln!(f, "{indent}{I}case {variant_i}: {{");
                swriteln!(f, "{indent}{I}{I}({value}).template emplace<{variant_i}>();");
                if let Some(payload_ty) = payload_ty {
                    emit_value_from_bits(
                        f,
                        indent.nest().nest(),
                        payload_ty,
                        &format!("std::get<{variant_i}>({value})"),
                        &format!("{offset} + {payload_offset}"),
                        load,
                        temp_name,
                    );
                }
                swriteln!(f, "{indent}{I}{I}break;");
                swriteln!(f, "{indent}{I}}}");
            }
            swriteln!(f, "{indent}}}");
        }
    }
}

pub fn offset_identifier(offset: &str) -> String {
    offset
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

const I: &str = Indent::I;
