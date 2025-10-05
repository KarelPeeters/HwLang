use crate::mid::ir::{IrModule, IrModules};
use crate::syntax::ast::PortDirection;
use crate::util::arena::IndexType;
use crate::util::data::IndexMapExt;
use hwl_util::swriteln;
use indexmap::{IndexMap, IndexSet};
use itertools::{Itertools, enumerate};
use regex::{Captures, Regex};
use std::convert::identity;
use std::num::NonZeroU16;

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
            // zero-width ports don't exist in verilog, and so can't be accessed here
            if port_info.ty.size_bits().is_zero() {
                continue;
            }

            // check if this port should get this accessor function
            let include_dir = match (port_info.direction, dir) {
                // input ports support set/get
                (PortDirection::Input, _) => true,
                // outputs ports only support get
                (PortDirection::Output, PortDirection::Output) => true,
                (PortDirection::Output, PortDirection::Input) => false,
            };
            if !include_dir {
                continue;
            }

            // add a case branch for the port
            let port_name = &port_info.name;
            let indent = if port_index == 0 { "" } else { "            " };
            swriteln!(
                f,
                "{indent}case {port_index}: return {prefix}_port_impl(wrapper->top->{port_name}, data_len, data);"
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
