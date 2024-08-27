use num_bigint::BigInt;

#[derive(Debug)]
pub struct SymbolicProblem {
    check: u32,
    bool_var_count: u32,
    int_var_count: u32,
    constraints: Vec<SymbolicBoolExpression>,
}

#[derive(Debug)]
pub enum SymbolicCheckOutcome {
    Satisfied(SymbolicSolution),
    Unsatisfiable,
    Unknown,
}

#[derive(Debug)]
pub struct SymbolicSolution {
    check: u32,
    values_bool: Vec<bool>,
    values_int: Vec<BigInt>,
}

#[derive(Debug)]
pub struct SymbolicBoolVar {
    check: u32,
    index_bool: u32,
}

#[derive(Debug)]
pub struct SymbolicIntVar {
    check: u32,
    index_int: u32,
}

#[derive(Debug)]
pub enum SymbolicIntExpression {
    Var(SymbolicIntVar)
}

#[derive(Debug)]
pub enum SymbolicBoolExpression {
    Var(SymbolicBoolVar)
}

impl SymbolicProblem {
    pub fn new() -> Self {
        Self {
            check: rand::random(),
            bool_var_count: 0,
            int_var_count: 0,
            constraints: vec![],
        }
    }

    pub fn add_bool_var(&mut self) -> SymbolicIntExpression {
        todo!()
    }

    pub fn add_int_var(&mut self) -> SymbolicIntVar {
        todo!()
    }

    pub fn add_constraint(&mut self, _condition: SymbolicBoolExpression) {
        todo!()
    }

    pub fn check(&self) -> SymbolicCheckOutcome {
        if self.constraints.is_empty() {
            SymbolicCheckOutcome::Satisfied(SymbolicSolution {
                check: self.check,
                values_bool: vec![false; self.bool_var_count as usize],
                values_int: vec![BigInt::ZERO; self.int_var_count as usize],
            })
        } else {
            SymbolicCheckOutcome::Unknown
        }
    }
}

impl SymbolicSolution {
    pub fn eval_bool(&self, expr: SymbolicBoolExpression) -> bool {
        match expr {
            SymbolicBoolExpression::Var(var) => {
                assert_eq!(var.check, self.check);
                self.values_bool[var.index_bool as usize]
            }
        }
    }

    pub fn eval_int(&self, expr: SymbolicIntExpression) -> BigInt {
        match expr {
            SymbolicIntExpression::Var(var) => {
                assert_eq!(var.check, self.check);
                self.values_int[var.index_int as usize].clone()
            }
        }
    }
}

impl SymbolicCheckOutcome {
    pub fn is_satisfied(&self) -> bool {
        matches!(self, SymbolicCheckOutcome::Satisfied(_))
    }

    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self, SymbolicCheckOutcome::Unsatisfiable)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, SymbolicCheckOutcome::Unknown)
    }
}