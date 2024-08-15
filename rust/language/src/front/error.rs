use num_bigint::BigInt;

use crate::front::driver::Item;
use crate::syntax::ast::{Args, Expression, Identifier, Path};

// TODO get rid of these error enums, instead directly report as printable construct
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
    
    InvalidPathStep(Identifier, Vec<String>),
    ExpectedPathToFile(Path),
    
    InvalidBuiltinIdentifier(Expression, Identifier),
    InvalidBuiltinArgs(Expression, Args),
    
    DuplicateParameterName(Identifier, Identifier),
    UnknownClock(Identifier),
}
