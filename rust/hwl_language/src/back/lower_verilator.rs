use crate::back::lower_verilog::LoweredVerilog;
use crate::mid::ir::{IrModule, IrModules};
use crate::syntax::ast::PortDirection;
use crate::util::data::IndexMapExt;
use fnv::FnvHasher;
use hwl_util::swriteln;
use indexmap::{IndexMap, IndexSet};
use itertools::{Itertools, enumerate};
use regex::{Captures, Regex};
use std::convert::identity;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct LoweredVerilator {
    pub source: String,
    pub top_class_name: String,
    pub check_hash: u64,
}

// TODO initialize ports to undefined or at least some valid value
/// Generate verilator C++ that wraps the top-level module into a dynamically linkable C API.
/// This can then be compiled and used by [super::wrap_verilator].
///
/// `source_hash` is used as an extra safety check to ensure that the correct module is being used at runtime,
/// can be any arbitrary value but needs to match the one passed to [super::wrap_verilator::VerilatedLib::new].
pub fn lower_verilator(modules: &IrModules, top_module: IrModule, verilog: &LoweredVerilog) -> LoweredVerilator {
    let top_module_info = &modules[top_module];

    const TEMPLATE: &str = include_str!("verilator_template.cpp");
    let mut replacements = IndexMap::new();

    // insert pert accessors
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
        replacements.insert_first(format!("PORTS_{}", prefix.to_uppercase()), f);
    }

    // insert check values
    // TODO hash generated C++?
    let check_hash = {
        let mut hasher = FnvHasher::default();
        verilog.source.hash(&mut hasher);
        verilog.top_module_name.hash(&mut hasher);
        TEMPLATE.hash(&mut hasher);
        hasher.finish()
    };
    replacements.insert_first("CHECK_HASH".to_owned(), format!("{check_hash}u"));

    // insert top class name
    const TOP_CLASS_NAME: &str = "VTop";
    replacements.insert_first("TOP_CLASS_NAME".to_owned(), TOP_CLASS_NAME.to_owned());

    let source = template_replace(TEMPLATE, &replacements).unwrap();
    LoweredVerilator {
        source,
        top_class_name: TOP_CLASS_NAME.to_owned(),
        check_hash,
    }
}

fn template_replace(template: &str, replacements: &IndexMap<String, String>) -> Result<String, String> {
    let regex = Regex::new("/\\*\\[TEMPLATE_(.+)]\\*/").unwrap();

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
