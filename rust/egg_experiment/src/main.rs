use egg::{rewrite, Rewrite, Runner, SymbolLang};
use itertools::Itertools;

fn main() {
    let mut rules: Vec<Rewrite<SymbolLang, ()>> = vec![];
    rules.push(rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"));
    rules.push(rewrite!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"));
    rules.extend(rewrite!("distribute-add-mul"; "(+ (* ?x ?y) (* ?x ?z))" <=> "(* ?x (+ ?y ?z))"));

    rules.push(rewrite!("add-0"; "(+ ?x 0)" => "?x"));
    rules.push(rewrite!("mul-0"; "(* ?x 0)" => "0"));
    rules.push(rewrite!("mul-1"; "(* ?x 1)" => "?x"));

    rules.push(rewrite!("reflex-eq"; "(= ?x ?x)" => "1"));
    rules.push(rewrite!("select-0"; "(s 0 ?x ?y)" => "?x"));
    rules.push(rewrite!("select-1"; "(s 1 ?x ?y)" => "?y"));

    let expr_false = "0".parse().unwrap();
    let expr_true = "1".parse().unwrap();
    let expr_test = "(s 1 0 1)".parse().unwrap();

    let mut runner = Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr_false)
        .with_expr(&expr_test)
        .with_expr(&expr_true)
        .run(&rules);
    runner.print_report();

    let is_true = !runner.egraph.equivs(&expr_test, &expr_true).is_empty();
    let is_false = !runner.egraph.equivs(&expr_test, &expr_false).is_empty();

    if is_true {
        println!("true equiv:");
        println!("{}", indent(&runner.egraph.explain_equivalence(&expr_test, &expr_true).get_flat_string()));
    }
    if is_false {
        println!("false equiv:");
        println!("{}", indent(&runner.egraph.explain_equivalence(&expr_test, &expr_false).get_flat_string()));
    }

    match (is_true, is_false) {
        (true, true) => println!("error: expression cannot be true and false at the same time"),
        (false, false) => println!("conclusion: unknown"),
        (true, false) => println!("conclusion: proven true"),
        (false, true) => println!("conclusion: proven false"),
    }

    // println!("{:?}", runner.egraph);

    // let extractor = Extractor::new(&runner.egraph, AstSize);
    // let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    // assert_eq!(best_expr, "a".parse().unwrap());
    // assert_eq!(best_cost, 1);
}

fn indent(s: &str) -> String {
    s.lines().map(|s| format!("    {s}")).join("\n")
}