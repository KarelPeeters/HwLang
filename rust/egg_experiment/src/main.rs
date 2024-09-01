use egg::{rewrite, Analysis, AstSize, Condition, DidMerge, Extractor, Id, Language, Rewrite, Runner, Subst, SymbolLang, Var};
use itertools::Itertools;
use std::cmp::min;
use std::str::FromStr;

fn main() {
    let mut rules: Vec<Rewrite<SymbolLang, _>> = vec![];

    rules.push(rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)" if build_check(1, "?x ?y")));
    rules.extend(rewrite!("assoc-add"; "(+ (+ ?x ?y) ?z)" <=> "(+ ?x (+ ?y ?z))" if build_check(2, "?x ?y ?z")));
    rules.push(rewrite!("cancel-add-neg"; "(+ ?x (- ?x))" => "0" if build_check(0, "")));
    rules.push(rewrite!("add-0"; "(+ ?x 0)" => "?x" if build_check(0, "?x")));

    rules.push(rewrite!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)" if build_check(1, "?x ?y")));
    rules.extend(rewrite!("assoc-mul"; "(* (* ?x ?y) ?z)" <=> "(* ?x (* ?y ?z))" if build_check(2, "?x ?y ?z")));
    rules.push(rewrite!("distr-add-mul-fwd"; "(+ (* ?x ?z) (* ?y ?z))" => "(* (+ ?x ?y) ?z)" if build_check(2, "?x ?y ?z")));
    rules.push(rewrite!("distr-add-mul-back"; "(* (+ ?x ?y) ?z)" => "(+ (* ?x ?z) (* ?y ?z))" if build_check(3, "?x ?y ?z ?z")));
    rules.push(rewrite!("mul-0"; "(* ?x 0)" => "0" if build_check(0, "")));
    rules.push(rewrite!("mul-1"; "(* ?x 1)" => "?x" if build_check(0, "?x")));

    let expr = "(+ (* a (+ a b)) (- (* a b)))";

    let expr = expr.parse().unwrap();
    let runner: Runner<SymbolLang, AstSizeAnalysis> = Runner::new(AstSizeAnalysis)
        .with_explanations_enabled()
        .with_expr(&expr)
        .with_iter_limit(usize::MAX)
        .with_node_limit(usize::MAX)
        .run(&rules);

    runner.print_report();
    println!();
    let egraph = runner.egraph;
    let extractor = Extractor::new(&egraph, AstSize);

    println!("Generated classes:");
    for class in egraph.classes().sorted_by_key(|c| c.id) {
        let (_, expr) = extractor.find_best(class.id);
        println!("  Id({}) size={} expr={}", class.id, class.data, expr);
    }
    println!();

    println!("Original expression: {} => {}", expr, extractor.find_best(egraph.lookup_expr(&expr).unwrap()).1);
}

type EGraph = egg::EGraph<SymbolLang, AstSizeAnalysis>;

struct AstSizeAnalysis;

fn node_size(egraph: &EGraph, s: &SymbolLang) -> u32 {
    let SymbolLang { op: _, children } = s;
    children.iter().copied().fold(1, |a, c| a + egraph[c].data)
}

// TODO try without analysis if this doesn't work
fn build_check(base: u32, vars: &str) -> impl Condition<SymbolLang, AstSizeAnalysis> {
    let vars = if vars.is_empty() {
        vec![]
    } else {
        vars.split(' ')
            .map(|v| Var::from_str(v).unwrap())
            .collect_vec()
    };

    struct ConditionImpl {
        base: u32,
        vars: Vec<Var>,
    }

    impl Condition<SymbolLang, AstSizeAnalysis> for ConditionImpl {
        fn check(&self, egraph: &mut egg::EGraph<SymbolLang, AstSizeAnalysis>, eclass: Id, subst: &Subst) -> bool {
            let min_size = egraph[eclass].data;

            let operand_sum = self.vars.iter()
                .map(|&v| egraph[*subst.get(v).unwrap()].data)
                .sum::<u32>();
            let new_size = self.base + operand_sum;

            // let expr = egraph.id_to_expr(eclass);
            // println!("checking Id({}) expr={} subst_id={:?} new_size={}", eclass, expr, subst, new_size);

            // new_size <= old_min_size
            // TODO also require at most twice as complex as the most complex expression originally was
            new_size <= min(2 + 2 * min_size, 32)
            // true
        }

        fn vars(&self) -> Vec<Var> {
            self.vars.clone()
        }
    }

    ConditionImpl { base, vars }
}

impl Analysis<SymbolLang> for AstSizeAnalysis {
    type Data = u32;

    fn make(egraph: &mut EGraph, s: &SymbolLang) -> Self::Data {
        node_size(egraph, s)
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let new = min(*a, b);
        let merge = DidMerge(new != *a, new != b);
        *a = new;
        merge
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let nodes = &mut egraph[id].nodes;

        // if equivalent to a leaf node, only keep leafs
        // TODO does this never reduce power? what if a node is equivalent to multiple leafs, can new ones be found?
        if nodes.iter().any(|n| n.is_leaf()) {
            nodes.retain(|n| n.is_leaf());
            return;
        }

        // only keep nodes that are simple enough
        // let nodes = &egraph[id].nodes;
        // let mut keep_mask = nodes.iter().map(|n| node_size(egraph, n) <= 16).collect_vec();

        // let nodes = &mut egraph[id].nodes;
        // let mut keep_iter = keep_mask.into_iter();
        // nodes.retain_mut(|n| keep_iter.next().unwrap());
        // assert_eq!(None, keep_iter.next());
    }
}
