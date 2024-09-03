use egg::{rewrite, Analysis, AstSize, Condition, CostFunction, DidMerge, Extractor, Id, Rewrite, Runner, Subst, SymbolLang, Var};
use itertools::Itertools;
use std::cmp::min;
use std::str::FromStr;
use std::time::Duration;

fn main() {
    let mut rules: Vec<Rewrite<SymbolLang, _>> = vec![];

    let expr = "(+ (* a (+ a b)) (- (* a b)))";
    let expr = expr.parse().unwrap();

    let max_ast_size = AstSize.cost_rec(&expr) * 2;
    let build_check = |base, vars| build_check(max_ast_size, base, vars);

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

    let runner: Runner<SymbolLang, AstSizeAnalysis> = Runner::new(AstSizeAnalysis)
        .with_explanations_enabled()
        .with_expr(&expr)
        .with_iter_limit(usize::MAX)
        .with_node_limit(usize::MAX)
        .with_time_limit(Duration::MAX)
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

fn build_check(max: usize, base: usize, vars: &str) -> impl Condition<SymbolLang, AstSizeAnalysis> {
    let vars = if vars.is_empty() {
        vec![]
    } else {
        vars.split(' ')
            .map(|v| Var::from_str(v).unwrap())
            .collect_vec()
    };

    struct ConditionImpl {
        base: usize,
        vars: Vec<Var>,
        max: usize,
    }

    impl Condition<SymbolLang, AstSizeAnalysis> for ConditionImpl {
        fn check(&self, egraph: &mut egg::EGraph<SymbolLang, AstSizeAnalysis>, eclass: Id, subst: &Subst) -> bool {
            let min_size = egraph[eclass].data;
            if min_size == 1 {
                return false;
            }

            let operand_sum = self.vars.iter()
                .map(|&v| egraph[*subst.get(v).unwrap()].data)
                .sum::<usize>();
            let new_size = self.base + operand_sum;

            new_size <= self.max
        }

        fn vars(&self) -> Vec<Var> {
            self.vars.clone()
        }
    }

    ConditionImpl { max, base, vars }
}

impl Analysis<SymbolLang> for AstSizeAnalysis {
    type Data = usize;

    fn make(egraph: &mut EGraph, s: &SymbolLang) -> Self::Data {
        let SymbolLang { op: _, children } = s;
        children.iter().copied().fold(1, |a, c| a + egraph[c].data)
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let new = min(*a, b);
        let merge = DidMerge(new != *a, new != b);
        *a = new;
        merge
    }
}
