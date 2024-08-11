use num_bigint::BigInt;
use crate::front::driver::Item;
use crate::syntax::ast::{Args, Expression, Identifier, Path};

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<Item>),

    ExpectedTypeExpressionGotValue(Expression),
    ExpectedValueExpressionGotType(Expression),
    ExpectedTypeExpressionGotConstructor(Expression),
    ExpectedValueExpressionGotConstructor(Expression),

    ExpectedFunctionExpression(Expression),
    ExpectedIntegerExpression(Expression),
    ExpectedRangeExpression(Expression),

    ExpectedNonNegativeInteger(Expression, BigInt),

    InvalidPathStep(Identifier, Vec<String>),
    ExpectedPathToFile(Path),

    InvalidBuiltinIdentifier(Expression, Identifier),
    InvalidBuiltinArgs(Expression, Args),

    DuplicateParameterName(Identifier, Identifier),
    UnknownClock(Identifier),
}
