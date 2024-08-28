use crate::expression::SymbolicBoolExpression;
use crate::expression::SymbolicIntExpression;
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

#[derive(Debug, Copy, Clone)]
pub struct SymbolicBoolVar {
    check: u32,
    index_bool: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct SymbolicIntVar {
    check: u32,
    index_int: u32,
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

    pub fn add_bool_var(&mut self) -> SymbolicBoolVar {
        let index_bool = self.bool_var_count;
        self.bool_var_count += 1;
        SymbolicBoolVar { check: self.check, index_bool }
    }

    pub fn add_int_var(&mut self) -> SymbolicIntVar {
        let index_int = self.int_var_count;
        self.int_var_count += 1;
        SymbolicIntVar { check: self.check, index_int }
    }

    pub fn add_constraint(&mut self, cond: SymbolicBoolExpression) {
        self.constraints.push(cond);
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
    pub fn eval_bool_var(&self, var: SymbolicBoolVar) -> bool {
        assert_eq!(var.check, self.check);
        self.values_bool[var.index_bool as usize]
    }

    pub fn eval_int_var(&self, var: SymbolicIntVar) -> BigInt {
        assert_eq!(var.check, self.check);
        self.values_int[var.index_int as usize].clone()
    }

    pub fn eval_bool(&self, expr: &SymbolicBoolExpression) -> bool {
        match expr {
            &SymbolicBoolExpression::Const(val) => val,
            &SymbolicBoolExpression::Var(var) => self.eval_bool_var(var),
            SymbolicBoolExpression::Not(cond) => !self.eval_bool(cond),
            SymbolicBoolExpression::And(lhs, rhs) => self.eval_bool(lhs) && self.eval_bool(rhs),
            SymbolicBoolExpression::Or(lhs, rhs) => self.eval_bool(lhs) || self.eval_bool(rhs),
            SymbolicBoolExpression::Xor(lhs, rhs) => self.eval_bool(lhs) ^ self.eval_bool(rhs),
            SymbolicBoolExpression::Lt(lhs, rhs) => self.eval_int(lhs) < self.eval_int(rhs),
            SymbolicBoolExpression::Eq(lhs, rhs) => self.eval_int(lhs) == self.eval_int(rhs),
        }
    }

    pub fn eval_int(&self, expr: &SymbolicIntExpression) -> BigInt {
        match expr {
            SymbolicIntExpression::Const(val) => val.clone(),
            &SymbolicIntExpression::Var(var) => self.eval_int_var(var),
            SymbolicIntExpression::Add(lhs, rhs) => self.eval_int(lhs) + self.eval_int(rhs),
            SymbolicIntExpression::Sub(lhs, rhs) => self.eval_int(lhs) - self.eval_int(rhs),
            SymbolicIntExpression::Mul(lhs, rhs) => self.eval_int(lhs) * self.eval_int(rhs),
            // TODO proper impl or Result
            SymbolicIntExpression::Pow(lhs, rhs) => self.eval_int(lhs).pow(self.eval_int(rhs).try_into().unwrap()),
            SymbolicIntExpression::Select(cond, lhs, rhs) => {
                // always evaluate both sides to check their validness
                let lhs = self.eval_int(lhs);
                let rhs = self.eval_int(rhs);
                if self.eval_bool(cond) { lhs } else { rhs }
            }
        }
    }
}

impl SymbolicCheckOutcome {
    pub fn is_satisfied(&self) -> bool {
        matches!(self, SymbolicCheckOutcome::Satisfied(_))
    }

    pub fn unwrap_satisfied(self) -> SymbolicSolution {
        match self {
            SymbolicCheckOutcome::Satisfied(solution) => solution,
            _ => panic!("Expected Satisfied, got {:?}", self),
        }
    }

    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self, SymbolicCheckOutcome::Unsatisfiable)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, SymbolicCheckOutcome::Unknown)
    }
}