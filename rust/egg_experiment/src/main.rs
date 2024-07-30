use egg::{Rewrite, SymbolLang, rewrite, Runner, Extractor, AstSize};

fn main() {
    let rules: &[Rewrite<SymbolLang, ()>] = &[
        rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"),
        rewrite!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"),

        rewrite!("distribute-add-mul"; "(+ (* ?x ?y) (* ?x ?z))" => "(* ?x (+ ?y ?z))"),
        
        //
        // rewrite!("add-0"; "(+ ?x 0)" => "?x"),
        // rewrite!("mul-0"; "(* ?x 0)" => "0"),
        // rewrite!("mul-1"; "(* ?x 1)" => "?x"),
    ];

    let expr_left = "(* (+ a b) c)".parse().unwrap();
    let expr_right = "(+ (* c b) (* a c))".parse().unwrap();

    let mut runner = Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr_left)
        .with_expr(&expr_right)
        .run(rules);
    runner.print_report();

    let eq = runner.egraph.equivs(&expr_left, &expr_right);
    println!("equivalence: {:?}", eq);
    if !eq.is_empty() {
        println!("{}", runner.egraph.explain_equivalence(&expr_left, &expr_right));
    }

    // println!("{:?}", runner.egraph);

    // let extractor = Extractor::new(&runner.egraph, AstSize);
    // let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    // assert_eq!(best_expr, "a".parse().unwrap());
    // assert_eq!(best_cost, 1);
}
