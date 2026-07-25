use crate::util::data::NonEmptyVec;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Find the set of strongly connected components (SCCs) on the given graph.
///
/// The graph is represented by the two parameters:
/// * `nodes` is a complete list of nodes without any duplicates.
/// * `children` should be a deterministic function that for each node returns all direct children.
///
/// Returns a vector where each item is a SCC.
///
/// To match the definition of SCC:
/// * Nodes that point to themselves don't affect the result in any way.
///   If identifying self-loops is important, the caller should still check for those separately.
/// * Nodes that are not part of any loops appear as single-node SCCs.
pub fn find_strongly_connected_components<T: Eq + Hash + Copy, C: IntoIterator<Item = T>>(
    nodes: impl IntoIterator<Item = T>,
    children: impl Fn(T) -> C,
) -> Vec<NonEmptyVec<T>> {
    // Implementation based on [Path-based strong component algorithm][1],
    //   with the callstack recursion implemented on the heap instead.
    // [1] Path-based strong component algorithm (https://en.wikipedia.org/wiki/Path-based_strong_component_algorithm)
    let mut node_to_number = HashMap::new();
    let mut node_has_component = HashSet::new();
    let mut component_to_nodes = vec![];
    let mut stack_p = vec![];
    let mut stack_s = vec![];

    // heap-allocated stack to avoid stack overflows on deep graphs
    let mut stack_frames: Vec<Frame<T, C::IntoIter>> = vec![];
    struct Frame<T, I> {
        node: T,
        children: I,
    }

    // populate initial stack
    for v in nodes {
        stack_frames.push(Frame {
            node: v,
            children: children(v).into_iter(),
        });
    }

    // repeatedly visit top stack frame
    while let Some(curr_frame) = stack_frames.last_mut() {
        let curr_node = curr_frame.node;

        // first time we visit this code?
        {
            let next_number = node_to_number.len();
            node_to_number.entry(curr_node).or_insert_with(|| {
                stack_s.push(curr_node);
                stack_p.push(curr_node);
                next_number
            });
        }

        if let Some(child_node) = curr_frame.children.next() {
            // visit next child
            if !node_to_number.contains_key(&child_node) {
                stack_frames.push(Frame {
                    node: child_node,
                    children: children(child_node).into_iter(),
                });
            } else if !node_has_component.contains(&child_node) {
                let child_number = node_to_number[&child_node];
                while let Some(&top_p) = stack_p.last() {
                    let top_p_number = node_to_number[&top_p];
                    if top_p_number <= child_number {
                        break;
                    }
                    stack_p.pop();
                }
            }
        } else {
            // all children have been visited, finish this node
            if stack_p.last() == Some(&curr_node) {
                let mut component_nodes = vec![];
                loop {
                    let node = stack_s.pop().unwrap();
                    assert!(node_has_component.insert(node));
                    component_nodes.push(node);
                    if node == curr_node {
                        break;
                    }
                }
                component_to_nodes.push(NonEmptyVec::try_from(component_nodes).unwrap());
                stack_p.pop();
            }
            stack_frames.pop();
        }
    }

    component_to_nodes
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
