use hwl_symbolic_prover::{SymbolicCheckOutcome, SymbolicProblem};

#[test]
fn empty() {
    let problem = SymbolicProblem::new();
    let outcome = problem.check();
    assert!(matches!(outcome, SymbolicCheckOutcome::Satisfied(_)), "{:?}", outcome);
}
