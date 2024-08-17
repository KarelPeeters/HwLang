use crate::back::LowerError::NoTopFileFound;
use crate::error::CompileError;
use crate::front::common::ScopedEntry;
use crate::front::diagnostic::DiagnosticAddable;
use crate::front::driver::{CompiledDataBase, Item};
use crate::front::scope::Visibility;
use crate::front::source::SourceDatabase;
use crate::front::types::{MaybeConstructor, Type};

#[derive(Debug)]
pub enum LowerError {
    NoTopFileFound,
}

// TODO make backend configurable between verilog and VHDL?
pub fn lower(source: &SourceDatabase, compiled: &CompiledDataBase) -> Result<LoweredDatabase, CompileError> {
    println!("finding top module");
    let _top_module = find_top_module(source, compiled)?;

    // TODO actual lowering
    //   start top-down, only lowering modules and maybe functions that are actually used

    Ok(LoweredDatabase {})
}

fn find_top_module(source: &SourceDatabase, compiled: &CompiledDataBase) -> Result<Item, CompileError> {
    let top_dir = *source[source.root_directory].children.get("top")
        .ok_or(LowerError::NoTopFileFound)?;
    let top_file = source[top_dir].file.ok_or(NoTopFileFound)?;
    let top_entry = &compiled[top_file].local_scope.find_immediate_str(source, "top", Visibility::Public)?;
    match top_entry.value {
        &ScopedEntry::Item(item) => {
            match compiled[item].ty.as_ref().unwrap() {
                MaybeConstructor::Constructor(_) => {
                    let err = source.diagnostic("top should be a module, got a constructor")
                        .add_error(top_entry.defining_span, "defined here")
                        .finish();
                    Err(err.into())
                }
                MaybeConstructor::Immediate(ty) => {
                    if let Type::Module(_) = ty {
                        Ok(item)
                    } else {
                        let err = source.diagnostic("top should be a module, got a non-module type")
                            .add_error(top_entry.defining_span, "defined here")
                            .finish();
                        Err(err.into())
                    }
                }
            }
        }
        ScopedEntry::Direct(_) => {
            // TODO include "got" string
            // TODO is this even ever possible? direct should only be inside of scopes
            let err = source.diagnostic("top should be an item, got a direct")
                .add_error(top_entry.defining_span, "defined here")
                .finish();
            Err(err.into())
        }
    }
}

pub struct LoweredDatabase {
    // TODO
}
