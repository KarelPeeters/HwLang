use std::marker::PhantomData;
use num_bigint::BigInt;
use crate::data::diagnostic::ErrorGuaranteed;

// TODO think about how many solver to instantiate
//    * we don't want a single program-wide solver, that would be very slow
//    * for now just nest the solver on each if condition, even though that maybe loses a bit of expression power
pub struct Solver<'p> {
    parent: PhantomData<&'p ()>,
    next_index: u64,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SolverAny {
    Bool(SolverBool),
    Int(SolverInt),
    Opaque(SolverOpaque),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct SolverBool(u64);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct SolverInt(u64);

// TODO doc: instances have to follow equality rules
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct SolverOpaque(u64);

#[derive(Debug, Copy, Clone)]
pub struct Unknown;

#[derive(Debug, Copy, Clone)]
pub struct UnknownOrFalse;

#[derive(Debug, Clone)]
struct Unique(u64);

#[derive(Debug, Clone)]
enum SolverBoolDef {
    // TODO considering removing this and keeping errors out of the solver
    Error(ErrorGuaranteed),

    Arbitrary(Unique),
    Known(bool),

    BoolAnd(Vec<SolverBool>),
    BoolNot(SolverBool),

    IntEq(SolverInt, SolverInt),
    IntLte(SolverInt, SolverInt),
}

#[derive(Debug, Clone)]
enum SolverIntDef {
    // TODO considering removing this and keeping errors out of the solver
    Error(ErrorGuaranteed),

    Arbitrary(Unique),
    Known(BigInt),

    Negate(SolverInt),
    Sum(Vec<SolverInt>),
    Product(Vec<SolverInt>),
}

// core functions
impl Solver<'_> {
    pub fn new() -> Solver<'static> {
        Solver { parent: PhantomData, next_index: 0 }
    }

    pub fn new_bool(&mut self, def: SolverBoolDef) -> SolverBool {
        // TODO deduplication
        // TODO check that vars in def belong to this solver
        let _ = def;
        let index = self.next_index;
        self.next_index += 1;
        SolverBool(index)
    }

    pub fn new_int(&mut self, def: SolverIntDef) -> SolverInt {
        // TODO deduplication
        // TODO check that vars in def belong to this solver
        let _ = def;
        let index = self.next_index;
        self.next_index += 1;
        SolverInt(index)
    }

    // TODO go through users of this function, and intern values that properly follow equality
    //   (after SSA value numbering, this will be all values)
    pub fn new_opaque(&mut self) -> SolverOpaque {
        // TODO deduplication
        // TODO check that vars in def belong to this solver
        let index = self.next_index;
        self.next_index += 1;
        SolverOpaque(index)
    }

    pub fn add_axiom(&mut self, axiom: SolverBool) {

        // TODO deduplication
        // TODO check that vars in def belong to this solver
        let _ = axiom;
    }

    pub fn eval_bool(&self, x: SolverBool) -> Result<Result<bool, Unknown>, ErrorGuaranteed> {
        // TODO check that vars in def belong to this solver
        let _ = x;
        Ok(Err(Unknown))
    }

    // TODO change this to instead be an error message
    pub fn eval_bool_true(&self, x: SolverBool) -> Result<Result<(), UnknownOrFalse>, ErrorGuaranteed> {
        // TODO return err?
        match self.eval_bool(x) {
            Ok(Ok(true)) => Ok(Ok(())),
            Ok(Ok(false)) => Ok(Err(UnknownOrFalse)),
            Ok(Err(Unknown)) => Ok(Err(UnknownOrFalse)),
            Err(e) => Err(e),
        }
    }

    pub fn nest(&self) -> Solver {
        todo!()
    }
}

// extra utility builder functions
impl Solver<'_> {
    pub fn known_bool(&mut self, x: bool) -> SolverBool {
        self.new_bool(SolverBoolDef::Known(x))
    }

    pub fn known_int(&mut self, x: impl Into<BigInt>) -> SolverInt {
        self.new_int(SolverIntDef::Known(x.into()))
    }

    pub fn arbitrary_bool(&mut self) -> SolverBool {
        self.new_bool(SolverBoolDef::Arbitrary(Unique(self.next_index)))
    }

    pub fn arbitrary_int(&mut self) -> SolverInt {
        self.new_int(SolverIntDef::Arbitrary(Unique(self.next_index)))
    }

    pub fn not(&mut self, b: SolverBool) -> SolverBool {
        self.new_bool(SolverBoolDef::BoolNot(b))
    }

    pub fn and(&mut self, a: SolverBool, b: SolverBool) -> SolverBool {
        self.new_bool(SolverBoolDef::BoolAnd(vec![a, b]))
    }

    pub fn or(&mut self, a: SolverBool, b: SolverBool) -> SolverBool {
        let a_not = self.not(a);
        let b_not = self.not(b);
        let and = self.and(a_not, b_not);
        self.not(and)
    }

    pub fn sum(&mut self, v: Vec<SolverInt>) -> SolverInt {
        self.new_int(SolverIntDef::Sum(v))
    }

    pub fn add(&mut self, a: SolverInt, b: SolverInt) -> SolverInt {
        self.sum(vec![a, b])
    }

    pub fn add_const(&mut self, a: SolverInt, b: impl Into<BigInt>) -> SolverInt {
        let b = self.known_int(b);
        self.add(a, b)
    }

    pub fn sub(&mut self, a: SolverInt, b: SolverInt) -> SolverInt {
        let b_neg = self.negate(b);
        self.add(a, b_neg)
    }

    pub fn zero(&mut self) -> SolverInt {
        self.known_int(0)
    }

    pub fn one(&mut self) -> SolverInt {
        self.known_int(1)
    }

    pub fn negate(&mut self, a: SolverInt) -> SolverInt {
        self.new_int(SolverIntDef::Negate(a))
    }

    pub fn compare_lte(&mut self, a: SolverInt, b: SolverInt) -> SolverBool {
        self.new_bool(SolverBoolDef::IntLte(a, b))
    }

    pub fn compare_lt(&mut self, a: SolverInt, b: SolverInt) -> SolverBool {
        let one = self.new_int(SolverIntDef::Known(BigInt::from(-1)));
        let b_minus_one = self.new_int(SolverIntDef::Sum(vec![b, one]));
        self.compare_lte(a, b_minus_one)
    }

    pub fn compare_gte(&mut self, a: SolverInt, b: SolverInt) -> SolverBool {
        self.compare_lte(b, a)
    }

    pub fn compare_gt(&mut self, a: SolverInt, b: SolverInt) -> SolverBool {
        self.compare_lt(b, a)
    }

    pub fn compare_eq(&mut self, a: SolverInt, b: SolverInt) -> SolverBool {
        self.new_bool(SolverBoolDef::IntEq(a, b))
    }

    pub fn error_bool(&mut self, e: ErrorGuaranteed) -> SolverBool {
        self.new_bool(SolverBoolDef::Error(e))
    }

    // TODO make SolverInt an enum with an error branch instead, same for bool
    pub fn error_int(&mut self, e: ErrorGuaranteed) -> SolverInt {
        self.new_int(SolverIntDef::Error(e))
    }
}