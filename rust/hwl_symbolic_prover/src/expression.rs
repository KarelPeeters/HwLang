use num_bigint::BigInt;
use crate::problem::{SymbolicBoolVar, SymbolicIntVar};

#[derive(Debug)]
pub enum SymbolicBoolExpression {
    // special
    Const(bool),
    Var(SymbolicBoolVar),

    // pure boolean
    Not(Box<SymbolicBoolExpression>),
    And(Box<SymbolicBoolExpression>, Box<SymbolicBoolExpression>),
    Or(Box<SymbolicBoolExpression>, Box<SymbolicBoolExpression>),
    Xor(Box<SymbolicBoolExpression>, Box<SymbolicBoolExpression>),

    // int->bool
    Lt(Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
    Eq(Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
}

#[derive(Debug)]
pub enum SymbolicIntExpression {
    // special
    Const(BigInt),
    Var(SymbolicIntVar),
    
    // pure int
    Add(Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
    Sub(Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
    Mul(Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
    Pow(Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
    
    // bool->int
    Select(Box<SymbolicBoolExpression>, Box<SymbolicIntExpression>, Box<SymbolicIntExpression>),
}

impl From<SymbolicBoolVar> for SymbolicBoolExpression {
    fn from(var: SymbolicBoolVar) -> Self {
        SymbolicBoolExpression::Var(var)
    }
}

impl From<bool> for SymbolicBoolExpression {
    fn from(val: bool) -> Self {
        SymbolicBoolExpression::Const(val)
    }
}

impl From<SymbolicIntVar> for SymbolicIntExpression {
    fn from(var: SymbolicIntVar) -> Self {
        SymbolicIntExpression::Var(var)
    }
}

impl From<BigInt> for SymbolicIntExpression {
    fn from(val: BigInt) -> Self {
        SymbolicIntExpression::Const(val)
    }
}
