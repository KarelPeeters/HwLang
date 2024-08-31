use egg::{rewrite, EGraph, Id, Rewrite, Runner, SymbolLang};
use itertools::Itertools;

fn main() {
    let rules = rules();

    // example: test whether assuming (a: uint; b: uint), (a..a+b) fits into uint
    //   known are:
    //   * a >= 0
    //   * b >= 0
    //   the things we need to check are:
    //   * a >= 0
    //   * a + b >= a

    // note: >= and other comparisons always have "0" as the right-hand-side
    let expr_known = vec![
        // "(>= a)",
        // "(>= b)",
        "(+ a (+ b (- a)))"
    ];
    let expr_to_prove = vec![
        // "(>= a)",
        // "(>= (+ (+ a b) (- a)))",
        // "(>= (** 2 a))",
    ];

    let mut egraph = EGraph::new(())
        .with_explanations_enabled();

    println!("Mapping expressions");
    let mut map_expr = |e: &&str| {
        let e = e.parse().unwrap();
        let id = egraph.add_expr(&e);
        println!("  {e} => Id({id})");
        id
    };
    let expr_known = expr_known.iter().map(&mut map_expr).collect_vec();
    let expr_to_prove = expr_to_prove.iter().map(&mut map_expr).collect_vec();

    println!("Running");
    let runner = Runner::default()
        .with_explanations_enabled()
        .with_egraph(egraph)
        .with_node_limit(32)
        .run(&rules);
    runner.print_report();
    let egraph = runner.egraph;

    println!("Resulting egraph:");
    for class in egraph.classes().sorted_by_key(|c| c.id) {
        print!("  Id({:?}) {}", class.id, egraph.id_to_expr(class.id));
        println!(":");
        for node in &class.nodes {
            println!("     {:?}", node);
        }
    }

    // egraph.dot().to_svg("ignored/egraph.svg").unwrap();

    println!("checking proof");
    let mut all_known_true = true;
    for expr_raw_id in expr_to_prove {
        let expr_id = egraph.find(expr_raw_id);
        let id_str = format!("Id({expr_raw_id}), Id({expr_id})");

        match check_true(&egraph, &expr_known, expr_id) {
            Check::KnownTrue => {
                println!("  proved {id_str}");
            }
            Check::KnownFalse => {
                println!("  failed to prove {id_str}, known false");
                all_known_true = false;
            }
            Check::Unknown => {
                println!("  failed to prove {id_str}, unknown");
                all_known_true = false;
            }
        }
    }
    if all_known_true {
        println!("=> success")
    } else {
        println!("=> failed")
    }

    // let is_true = !runner.egraph.equivs(&expr_test, &expr_true).is_empty();
    // let is_false = !runner.egraph.equivs(&expr_test, &expr_false).is_empty();
    //
    // if is_true {
    //     println!("true equiv:");
    //     println!("{}", indent(&runner.egraph.explain_equivalence(&expr_test, &expr_true).get_flat_string()));
    // }
    // if is_false {
    //     println!("false equiv:");
    //     println!("{}", indent(&runner.egraph.explain_equivalence(&expr_test, &expr_false).get_flat_string()));
    // }
    //
    // match (is_true, is_false) {
    //     (true, true) => println!("error: expression cannot be true and false at the same time"),
    //     (false, false) => println!("conclusion: unknown"),
    //     (true, false) => println!("conclusion: proven true"),
    //     (false, true) => println!("conclusion: proven false"),
    // }

    // println!("{:?}", runner.egraph);

    // let extractor = Extractor::new(&runner.egraph, AstSize);
    // let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);
    // assert_eq!(best_expr, "a".parse().unwrap());
    // assert_eq!(best_cost, 1);
}

enum Check {
    KnownTrue,
    // TODO find counterexample that fits all known conditions?
    //   it's a lot simpler if we only allow input variable ranges as known conditions...
    KnownFalse,
    Unknown,
}

fn check_true(_graph: &EGraph<SymbolLang, ()>, known: &[Id], to_prove: Id) -> Check {
    match known.contains(&to_prove) {
        true => Check::KnownTrue,
        false => Check::Unknown,
    }
}

fn rules() -> Vec<Rewrite<SymbolLang, ()>> {
    let mut rules: Vec<Rewrite<SymbolLang, ()>> = vec![];
    rules.push(rewrite!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"));
    rules.push(rewrite!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"));
    rules.extend(rewrite!("assoc-add"; "(+ (+ ?x ?y) ?z)" <=> "(+ ?x (+ ?y ?z))"));
    rules.extend(rewrite!("assoc-mul"; "(* (* ?x ?y) ?z)" <=> "(* ?x (* ?y ?z))"));
    rules.extend(rewrite!("distr-add-mul"; "(* ?x (+ ?y ?z))" <=> "(+ (* ?x ?y) (* ?x ?z))"));

    rules.push(rewrite!("cancel-add-neg"; "(+ ?x (- ?x))" => "0"));

    rules.push(rewrite!("add-0"; "(+ ?x 0)" => "?x"));
    rules.push(rewrite!("mul-0"; "(* ?x 0)" => "0"));
    rules.push(rewrite!("mul-1"; "(* ?x 1)" => "?x"));
    rules.push(rewrite!("neg-0"; "(- 0)" => "0"));

    rules.push(rewrite!("select-0"; "(s 0 ?x ?y)" => "?x"));
    rules.push(rewrite!("select-1"; "(s 1 ?x ?y)" => "?y"));
    rules.push(rewrite!("select-same"; "(s ?x ?y ?y)" => "?y"));

    rules
}

fn indent(s: &str) -> String {
    s.lines().map(|s| format!("    {s}")).join("\n")
}
