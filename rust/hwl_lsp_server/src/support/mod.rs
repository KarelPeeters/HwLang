pub mod find_definition;
pub mod find_usages;

#[cfg(test)]
mod find_definition_test;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PosNotOnIdentifier;
