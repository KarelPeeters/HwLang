use crate::data::compiled::{CompiledDatabase, Item};
use crate::data::parsed::ParsedDatabase;
use crate::front::types::GenericArguments;
use crate::syntax::ast;
use crate::syntax::ast::Identifier;
use crate::util::data::IndexMapExt;
use indexmap::{IndexMap, IndexSet};
use std::collections::VecDeque;
use std::fmt::Display;
use unwrap_match::unwrap_match;

pub struct BackModuleList {
    todo: VecDeque<(BackModuleName, BackModule)>,
    name: IndexMap<BackModule, BackModuleName>,
    used_names: IndexSet<String>,
}

#[derive(Clone, Debug)]
pub struct BackModuleName(String);

impl Display for BackModuleName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BackModule {
    pub item: Item,
    /// These args are constant and fully evaluated, without any remaining outer generic parameters.
    pub args: Option<GenericArguments>,
}

impl BackModuleList {
    pub fn new() -> Self {
        BackModuleList {
            todo: VecDeque::new(),
            name: IndexMap::new(),
            used_names: IndexSet::new(),
        }
    }

    pub fn push(&mut self, parsed: &ParsedDatabase, compiled: &CompiledDatabase, module: BackModule) -> BackModuleName {
        if let Some(name) = self.name.get(&module) {
            return name.clone();
        }

        let ast = parsed.item_ast(compiled[module.item].ast_ref);
        let ast = unwrap_match!(ast, ast::Item::Module(ast) => ast);
        let name = pick_unique_name(&ast.id, &mut self.used_names);

        let name = BackModuleName(name.clone());
        self.todo.push_front((name.clone(), module.clone()));
        self.name.insert_first(module, name.clone());
        name
    }

    pub fn pop(&mut self) -> Option<(BackModuleName, BackModule)> {
        self.todo.pop_front()
    }

    pub fn finish(self) -> IndexMap<BackModule, BackModuleName> {
        assert!(self.todo.is_empty());
        self.name
    }
}

// TODO proper filename uniqueness scheme: combination of path, raw name and generic args
//   it might be worth getting a bit clever about this, or we can also just always use the full name
//   but using the paths generic args fully might generate _very_ long strings
fn pick_unique_name(id: &Identifier, used_names: &mut IndexSet<String>) -> String {
    let raw_name = &id.string;

    if used_names.insert(raw_name.to_owned()) {
        // immediate success
        raw_name.to_owned()
    } else {
        // append suffix
        for i in 0.. {
            let cand = format!("{}_{}", raw_name, i);
            if used_names.insert(cand.clone()) {
                return cand;
            }
        }
        unreachable!()
    }
}