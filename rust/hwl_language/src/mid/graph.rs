use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::mid::ir::{IrModule, IrModuleChild, IrModules};
use crate::util::data::NonEmptyVec;
use indexmap::IndexSet;
use itertools::Itertools;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub fn ir_modules_topological_sort(modules: &IrModules, top: impl IntoIterator<Item = IrModule>) -> Vec<IrModule> {
    let mut seen = IndexSet::new();
    let mut todo = top.into_iter().collect_vec();

    while let Some(module) = todo.pop() {
        if !seen.insert(module) {
            continue;
        }

        for child in &modules[module].children {
            match &child.inner {
                IrModuleChild::ModuleInternalInstance(inst) => {
                    todo.push(inst.module);
                }
                IrModuleChild::ModuleExternalInstance(_)
                | IrModuleChild::ClockedProcess(_)
                | IrModuleChild::CombinatorialProcess(_) => {}
            }
        }
    }

    seen.into_iter().rev().collect_vec()
}

pub fn ir_modules_check_no_cycles(diags: &Diagnostics, modules: &IrModules) -> DiagResult {
    // find connected components
    let mut components = find_strongly_connected_components(modules.keys(), |module| {
        modules[module].children.iter().filter_map(|c| match &c.inner {
            IrModuleChild::ModuleInternalInstance(c) => Some(c.module),
            IrModuleChild::ClockedProcess(_)
            | IrModuleChild::CombinatorialProcess(_)
            | IrModuleChild::ModuleExternalInstance(_) => None,
        })
    });

    // keep only non-trivial components
    components.retain(|c| c.len() > 1);

    // sort to ensure deterministic diagnostics
    components
        .iter_mut()
        .for_each(|c| c.sort_by_key(|&m| modules[m].debug_info_id.span));
    components.sort_by_key(|c| modules[*c.first()].debug_info_id.span);

    // report diagnostics
    let mut any_err = Ok(());
    for c in components {
        any_err = Err(component_to_diagnostic(modules, c).report(diags));
    }
    any_err
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

fn find_strongly_connected_components<T: Eq + Hash + Copy, C: IntoIterator<Item = T>>(
    nodes: impl IntoIterator<Item = T>,
    children: impl Fn(T) -> C,
) -> Vec<NonEmptyVec<T>> {
    // Path-based strong component algorithm (https://en.wikipedia.org/wiki/Path-based_strong_component_algorithm)
    let mut node_to_number = HashMap::new();
    let mut node_has_component = HashSet::new();
    let mut component_to_nodes = vec![];
    let mut stack_p = vec![];
    let mut stack_s = vec![];

    for v in nodes {
        let _ = find_strongly_connected_components_visit(
            &children,
            &mut node_to_number,
            &mut node_has_component,
            &mut component_to_nodes,
            &mut stack_p,
            &mut stack_s,
            v,
        );
    }

    component_to_nodes
}

#[must_use]
fn find_strongly_connected_components_visit<T: Eq + Hash + Copy, C: IntoIterator<Item = T>>(
    children: &impl Fn(T) -> C,
    node_to_number: &mut HashMap<T, usize>,
    node_has_component: &mut HashSet<T>,
    component_to_nodes: &mut Vec<NonEmptyVec<T>>,
    stack_p: &mut Vec<T>,
    stack_s: &mut Vec<T>,
    v: T,
) -> bool {
    // Set the preorder number of v to C, and increment C.
    let c = node_to_number.len();
    match node_to_number.entry(v) {
        Entry::Occupied(_) => {
            // this is not the first visit of this node
            return false;
        }
        Entry::Vacant(e) => {
            e.insert(c);
        }
    }

    // Push v onto S and also onto P.
    stack_s.push(v);
    stack_p.push(v);

    // For each edge from v to a neighboring vertex w:
    for w in children(v) {
        // If the preorder number of w has not yet been assigned (the edge is a tree edge),
        //   recursively search w;
        let was_first_visit = find_strongly_connected_components_visit(
            children,
            node_to_number,
            node_has_component,
            component_to_nodes,
            stack_p,
            stack_s,
            w,
        );

        // Otherwise, if w has not yet been assigned to a strongly connected component
        //   (the edge is a forward/back/cross edge):
        if !was_first_visit && !node_has_component.contains(&w) {
            // Repeatedly pop vertices from P until the top element of P
            //   has a preorder number less than or equal to the preorder number of w.
            let w_number = node_to_number[&w];
            loop {
                if let Some(top_p) = stack_p.last() {
                    let top_p_number = node_to_number[top_p];
                    if top_p_number <= w_number {
                        break;
                    } else {
                        stack_p.pop();
                    }
                }
            }
        }
    }

    // If v is the top element of P:
    if Some(&v) == stack_p.last() {
        // Pop vertices from S until v has been popped, and assign the popped vertices to a new component.
        let mut component_nodes = vec![];

        loop {
            let w = stack_s.pop().unwrap();

            assert!(node_has_component.insert(w));
            component_nodes.push(w);

            if w == v {
                break;
            }
        }

        component_to_nodes.push(NonEmptyVec::try_from(component_nodes).unwrap());

        // Pop v from P.
        stack_p.pop();
    }

    // this was the first visit of this node
    true
}
