use crate::front::scope::Scope;
use crate::new::misc::TypeOrValue;
use crate::new::value::CompileValue;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionValue {
    pub outer_scope: Scope,
    // TODO (named) args
    // TODO return type
    pub body: FunctionBody,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum FunctionBody {
    Type,
    Enum(/*TODO*/),
    Struct(/*TODO*/),
    // TODO add normal functions
}

// TODO implement call_runtime which generates ir code
impl FunctionValue {
    pub fn call_compile_time(&self/*TODO args*/) -> TypeOrValue<CompileValue> {
        let _ = self.body;
        // TODO create scope, fill in args
        // TODO run body
        // TODO check return type
        // TODO actually return type
        todo!()
    }
}