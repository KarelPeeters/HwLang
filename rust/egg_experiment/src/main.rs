use egg::{rewrite, Rewrite, Runner, SymbolLang};
use itertools::Itertools;
use std::fmt::Write;

fn main() {
    // env_logger::init();
    
    let rules = rules();

    // for limit in [usize::MAX] {
    //     println!("limit = {limit}");
    let expr = "(+ a (+ b (- a)))";
    let runner = Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr.parse().unwrap())
        .with_node_limit(1024)
        .run(&rules);

    let mut s = String::new();
    let f = &mut s;
    writeln!(f, "{}", runner.report().to_string()).unwrap();

    writeln!(f, "Resulting egraph:").unwrap();
    let egraph = runner.egraph;
    for class in egraph.classes().sorted_by_key(|c| c.id) {
        write!(f, "  Id({:?}) {}", class.id, egraph.id_to_expr(class.id)).unwrap();
        writeln!(f, ":").unwrap();
        for node in &class.nodes {
            writeln!(f, "     {:?}", node).unwrap();
        }
    }

    // std::fs::write(format!("ignored/egraph_{limit:04}.txt"), s).unwrap();
}


fn rules() -> Vec<Rewrite<SymbolLang, ()>> {
    let mut rules: Vec<Rewrite<SymbolLang, ()>> = vec![];
    rules.push(rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"));
    // rules.push(rewrite!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"));
    rules.extend(rewrite!("assoc-add"; "(+ (+ ?x ?y) ?z)" <=> "(+ ?x (+ ?y ?z))"));
    // rules.extend(rewrite!("assoc-mul"; "(* (* ?x ?y) ?z)" <=> "(* ?x (* ?y ?z))"));
    // rules.extend(rewrite!("distr-add-mul"; "(* ?x (+ ?y ?z))" <=> "(+ (* ?x ?y) (* ?x ?z))"));

    rules.push(rewrite!("cancel-add-neg"; "(+ ?x (- ?x))" => "0"));

    rules.push(rewrite!("add-0"; "(+ ?x 0)" => "?x"));
    // rules.push(rewrite!("mul-0"; "(* ?x 0)" => "0"));
    // rules.push(rewrite!("mul-1"; "(* ?x 1)" => "?x"));
    // rules.push(rewrite!("neg-0"; "(- 0)" => "0"));

    // rules.push(rewrite!("select-0"; "(s 0 ?x ?y)" => "?x"));
    // rules.push(rewrite!("select-1"; "(s 1 ?x ?y)" => "?y"));
    // rules.push(rewrite!("select-same"; "(s ?x ?y ?y)" => "?y"));

    rules
}

