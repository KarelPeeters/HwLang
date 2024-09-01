use egg::{rewrite, Rewrite, Runner, SymbolLang};
use itertools::Itertools;

fn main() {
    let mut rules: Vec<Rewrite<SymbolLang, ()>> = vec![];
    rules.push(rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"));
    rules.extend(rewrite!("assoc-add"; "(+ (+ ?x ?y) ?z)" <=> "(+ ?x (+ ?y ?z))"));
    rules.push(rewrite!("cancel-add-neg"; "(+ ?x (- ?x))" => "0"));
    rules.push(rewrite!("add-0"; "(+ ?x 0)" => "?x"));

    let expr = "(+ a (+ b (- a)))";
    let runner = Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr.parse().unwrap())
        .with_node_limit(64)
        .run(&rules);

    runner.print_report();
    let mut egraph = runner.egraph;

    println!("Generated classes:");
    for class in egraph.classes().sorted_by_key(|c| c.id) {
        println!("  {}", egraph.id_to_expr(class.id));
    }

    println!("Explanation for final generated class:");
    let newest_id = egraph.classes().map(|c| c.id).max().unwrap();
    let newest_expr = egraph.id_to_expr(newest_id);
    println!("{}", egraph.explain_existance(&newest_expr).get_flat_string());
}
