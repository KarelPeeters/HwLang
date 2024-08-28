use hwl_symbolic_prover::expression::SymbolicBoolExpression;
use hwl_symbolic_prover::problem::SymbolicProblem;
use num_bigint::BigInt;

#[test]
fn empty() {
    let problem = SymbolicProblem::new();
    let outcome = problem.check();
    assert!(outcome.is_satisfied(), "{:?}", outcome);
}

#[test]
fn eq() {
    let mut problem = SymbolicProblem::new();
    let x = problem.add_int_var();
    problem.add_constraint(SymbolicBoolExpression::Eq(
        Box::new(x.into()),
        Box::new(BigInt::from(4u32).into()),
    ));
    let outcome = problem.check();
    let solution = outcome.unwrap_satisfied();
    assert_eq!(solution.eval_int_var(x), BigInt::from(4))
}

#[test]
fn range() {
    let mut problem = SymbolicProblem::new();
    let x = problem.add_int_var();
    problem.add_constraint(SymbolicBoolExpression::Lt(
        Box::new(x.into()),
        Box::new(BigInt::from(4u32).into()),
    ));
    problem.add_constraint(SymbolicBoolExpression::Lt(
        Box::new(BigInt::from(2u32).into()),
        Box::new(x.into()),
    ));

    let outcome = problem.check();
    let solution = outcome.unwrap_satisfied();
    assert_eq!(solution.eval_int_var(x), BigInt::from(3))
}

// TODO more test cases:
//  * multiple variables that interact
//  * power-of-two minus one issue
