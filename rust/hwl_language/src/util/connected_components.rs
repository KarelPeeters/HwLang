use crate::util::data::NonEmptyVec;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

// TODO should we distinguish between non-self loops and self loops here? all downstream users will care
// TODO maybe immediately build a scheduling graph at the same time?
pub fn find_strongly_connected_components<T: Eq + Hash + Copy, C: IntoIterator<Item = T>>(
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

#[cfg(test)]
mod tests {
    use crate::util::connected_components::find_strongly_connected_components;
    use crate::util::data::NonEmptyVec;
    use itertools::{Itertools, enumerate};

    fn test_case(graph: &[&[usize]], expected: &[&[usize]]) {
        let result = find_strongly_connected_components(0..graph.len(), |i| graph[i].iter().copied());

        // basic check: every node must appear in exactly one group
        let mut node_to_group: Vec<Option<usize>> = vec![None; graph.len()];
        for (group_index, group) in enumerate(&result) {
            for &node in group {
                let slot = &mut node_to_group[node];
                assert!(slot.is_none());
                *slot = Some(group_index);
            }
        }
        for slot in node_to_group {
            assert!(slot.is_some());
        }

        // check that the result matches what we expect
        let expected_sorted = expected
            .iter()
            .map(|x| NonEmptyVec::try_from(x.iter().copied().sorted().collect_vec()).unwrap())
            .sorted()
            .collect_vec();
        let result_sorted = {
            let mut result_sorted = result;
            for s in &mut result_sorted {
                s.sort();
            }
            result_sorted.sort();
            result_sorted
        };

        assert_eq!(result_sorted, expected_sorted);
    }

    #[test]
    fn trivial_graphs() {
        test_case(&[], &[]);

        test_case(&[&[]], &[&[0]]);
        test_case(&[&[], &[]], &[&[0], &[1]]);
    }

    #[test]
    fn non_loop_children() {
        test_case(&[&[1], &[]], &[&[0], &[1]]);
        test_case(&[&[1], &[2], &[]], &[&[0], &[1], &[2]]);
    }

    #[test]
    fn self_loop() {
        test_case(&[&[0]], &[&[0]]);
        test_case(&[&[0], &[1]], &[&[0], &[1]]);
    }

    #[test]
    fn basic_loops() {
        test_case(&[&[1], &[0]], &[&[0, 1]]);
        test_case(&[&[1], &[2], &[0]], &[&[0, 1, 2]]);
    }

    #[test]
    fn real_loops() {
        // some cases transcribed from examples online
        test_case(
            &[&[3, 7], &[0, 6], &[3], &[5], &[0, 8], &[9], &[8], &[5, 6], &[7], &[2]],
            &[&[0], &[1], &[2, 3, 5, 9], &[4], &[6, 7, 8]],
        );
        test_case(
            &[&[1], &[2, 4], &[3], &[0], &[5], &[6], &[4, 7], &[]],
            &[&[0, 1, 2, 3], &[4, 5, 6], &[7]],
        );
        test_case(
            &[
                &[1, 7],
                &[1, 2],
                &[1, 5],
                &[2, 4],
                &[9],
                &[3, 6, 9],
                &[2],
                &[0, 6],
                &[6],
                &[4],
            ],
            &[&[0, 7], &[1, 2, 3, 5, 6], &[4, 9], &[8]],
        )
    }
}
