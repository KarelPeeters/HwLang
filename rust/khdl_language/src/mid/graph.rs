use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::mid::ir::{IrDatabase, IrModule, IrModuleChild, IrModules};
use crate::util::connected_components::find_strongly_connected_components;
use crate::util::data::NonEmptyVec;
use crate::util::iter::IterExt;
use indexmap::IndexSet;
use itertools::Itertools;

impl IrDatabase {
    pub fn new_check_cycles(
        diags: &Diagnostics,
        modules: IrModules,
        external_modules: IndexSet<String>,
    ) -> DiagResult<Self> {
        let modules_sorted = ir_modules_sort_check_cycles(diags, &modules)?;
        Ok(IrDatabase {
            modules,
            external_modules,
            modules_sorted,
        })
    }

    /// Returns all modules reachable from `top`, sorted in reverse topological order.
    /// Child modules will appear before they are instantiated.
    pub fn modules_sorted_reachable_from(
        &self,
        top: impl IntoIterator<Item = IrModule>,
    ) -> impl Iterator<Item = IrModule> {
        let mut reachable = IndexSet::new();
        let mut todo = top.into_iter().collect_vec();
        while let Some(module) = todo.pop() {
            if reachable.insert(module) {
                todo.extend(module_child_modules(&self.modules, module));
            }
        }

        self.modules_sorted
            .iter()
            .copied()
            .filter(move |module| reachable.contains(module))
    }
}

fn ir_modules_sort_check_cycles(diags: &Diagnostics, modules: &IrModules) -> DiagResult<Vec<IrModule>> {
    // find connected components
    let components = find_strongly_connected_components(modules.keys(), |module| module_child_modules(modules, module));

    // split into properly sorted modules and cycles
    let mut sorted = vec![];
    let mut cycles = vec![];
    for c in components {
        if let Some(&m) = c.iter().single()
            && !module_child_modules(modules, m).contains(&m)
        {
            sorted.push(m);
        } else {
            cycles.push(c);
        }
    }

    // sort to ensure deterministic diagnostics
    // TODO might not be enough in case of generic modules, which get the same span
    cycles
        .iter_mut()
        .for_each(|c| c.sort_by_key(|&m| modules[m].debug_info_id.span));
    cycles.sort_by_key(|c| modules[*c.first()].debug_info_id.span);

    // report diagnostics
    let mut any_err = Ok(());
    for c in cycles {
        any_err = Err(component_to_diagnostic(modules, c).report(diags));
    }
    any_err?;

    // success
    Ok(sorted)
}

fn module_child_modules(modules: &IrModules, module: IrModule) -> impl Iterator<Item = IrModule> + '_ {
    modules[module].children.iter().filter_map(|c| match &c.inner {
        IrModuleChild::ModuleInternalInstance(c) => Some(c.module),
        IrModuleChild::ClockedProcess(_)
        | IrModuleChild::CombinatorialProcess(_)
        | IrModuleChild::ModuleExternalInstance(_) => None,
    })
}

fn component_to_diagnostic(modules: &IrModules, component: NonEmptyVec<IrModule>) -> DiagnosticError {
    let mut messages = vec![];
    for &parent in &component {
        let parent_info = &modules[parent];
        messages.push((parent_info.debug_info_id.span, "module declared here".to_owned()));

        for child in &parent_info.children {
            match &child.inner {
                IrModuleChild::ModuleInternalInstance(child_inner) => {
                    if component.contains(&child_inner.module) {
                        messages.push((child.span, "child instantiated here".to_owned()));
                    }
                }
                IrModuleChild::ClockedProcess(_)
                | IrModuleChild::CombinatorialProcess(_)
                | IrModuleChild::ModuleExternalInstance(_) => continue,
            }
        }
    }
    let messages = NonEmptyVec::try_from(messages).unwrap();

    DiagnosticError::new_multiple("cyclic module instantiation", messages)
}
