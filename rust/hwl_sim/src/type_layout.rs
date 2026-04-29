use hwl_language::sim::recorder::{WaveSignalType, enum_tag_width};

pub struct CompositeChild {
    pub name: String,
    pub ty: WaveSignalType,
    pub bit_offset: usize,
    pub bit_len: usize,
}

pub fn composite_children(ty: &WaveSignalType) -> Vec<CompositeChild> {
    let mut result = Vec::new();
    match ty {
        WaveSignalType::Array { len, element } => {
            let stride = element.bit_len();
            for i in 0..*len {
                result.push(CompositeChild {
                    name: format!("[{i}]"),
                    ty: element.as_ref().clone(),
                    bit_offset: i * stride,
                    bit_len: stride,
                });
            }
        }
        WaveSignalType::Tuple(elements) => {
            let mut bit_offset = 0;
            for (i, element) in elements.iter().enumerate() {
                let bit_len = element.bit_len();
                result.push(CompositeChild {
                    name: format!(".{i}"),
                    ty: element.clone(),
                    bit_offset,
                    bit_len,
                });
                bit_offset += bit_len;
            }
        }
        WaveSignalType::Struct { fields, .. } => {
            let mut bit_offset = 0;
            for (name, element) in fields {
                let bit_len = element.bit_len();
                result.push(CompositeChild {
                    name: format!(".{name}"),
                    ty: element.clone(),
                    bit_offset,
                    bit_len,
                });
                bit_offset += bit_len;
            }
        }
        WaveSignalType::Enum { variants, .. } => {
            let tag_width = enum_tag_width(variants.len());
            result.push(CompositeChild {
                name: ".tag".to_owned(),
                ty: WaveSignalType::Int {
                    signed: false,
                    width: tag_width,
                },
                bit_offset: 0,
                bit_len: tag_width,
            });
            for (name, payload) in variants {
                if let Some(payload) = payload {
                    let bit_len = payload.bit_len();
                    result.push(CompositeChild {
                        name: format!(".{name}"),
                        ty: payload.clone(),
                        bit_offset: tag_width,
                        bit_len,
                    });
                }
            }
        }
        _ => {}
    }
    result
}
