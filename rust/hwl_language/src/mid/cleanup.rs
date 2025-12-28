use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrDatabase, IrExpression, IrForStatement,
    IrIfStatement, IrLargeArena, IrModuleChild, IrModuleInfo, IrSignalOrVariable, IrStatement, IrStringPiece,
    IrStringSubstitution, IrTargetStep, IrVariable, IrVariables, ValueAccess,
};
use indexmap::{IndexMap, IndexSet};
use itertools::chain;

// TODO also remove signal->var copies
pub fn cleanup(db: &mut IrDatabase) {
    let IrDatabase {
        top_module: _,
        modules,
        external_modules: _,
    } = db;
    for (_, m) in modules {
        cleanup_module(m);
    }
}

pub fn cleanup_module(ir: &mut IrModuleInfo) {
    let IrModuleInfo {
        ports: _,
        registers: _,
        wires: _,
        large,
        children,
        debug_info_file: _,
        debug_info_id: _,
        debug_info_generic_args: _,
    } = ir;

    for child in children {
        match &mut child.inner {
            IrModuleChild::ClockedProcess(proc) => {
                let IrClockedProcess {
                    locals,
                    async_reset,
                    clock_signal: _,
                    clock_block,
                } = proc;

                // we don't need to worry about this, it is not allowed to access variables
                let _ = async_reset;

                cleanup_process(large, locals, clock_block);
            }
            IrModuleChild::CombinatorialProcess(proc) => {
                let IrCombinatorialProcess { locals, block } = proc;
                cleanup_process(large, locals, block);
            }
            IrModuleChild::ModuleInternalInstance(_) => {}
            IrModuleChild::ModuleExternalInstance(_) => {}
        }
    }
}

fn cleanup_process(large: &mut IrLargeArena, locals: &mut IrVariables, block: &mut IrBlock) {
    inline_vars_process(large, block);

    let mut used_vars = IndexSet::new();
    collect_used_vars_block(large, &mut used_vars, block);

    remove_dead_vars_block(block, &used_vars);
    locals.retain(|var, _| used_vars.contains(&var));
}

fn collect_used_vars_block(large: &IrLargeArena, used: &mut IndexSet<IrVariable>, block: &IrBlock) {
    block.visit_values_accessed(large, &mut |var, access| match access {
        ValueAccess::Read => {
            if let IrSignalOrVariable::Variable(var) = var {
                used.insert(var);
            }
        }
        ValueAccess::Write => {}
    });
}

fn remove_dead_vars_block(block: &mut IrBlock, used: &IndexSet<IrVariable>) {
    let IrBlock { statements } = block;
    statements.retain_mut(|stmt| match &mut stmt.inner {
        IrStatement::Assign(target, _source) => {
            let IrAssignmentTarget { base, steps: _ } = target;
            if let IrSignalOrVariable::Variable(base) = *base {
                used.contains(&base)
            } else {
                true
            }
        }
        IrStatement::Block(block) => {
            remove_dead_vars_block(block, used);
            !block.statements.is_empty()
        }
        IrStatement::If(IrIfStatement {
            condition: _,
            then_block,
            else_block,
        }) => {
            remove_dead_vars_block(then_block, used);
            if let Some(else_block) = else_block {
                remove_dead_vars_block(else_block, used);
            }
            true
        }
        IrStatement::For(IrForStatement {
            index: _,
            range: _,
            block,
        }) => {
            remove_dead_vars_block(block, used);
            true
        }
        IrStatement::Print(_) => true,
    })
}

struct VarState<'p> {
    parent: Option<&'p VarState<'p>>,
    map: IndexMap<IrVariable, VarInfo>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct VarInfo {
    version: u64,
    copy_of: Option<(IrVariable, u64)>,
}

fn inline_vars_process(large: &mut IrLargeArena, block: &mut IrBlock) {
    let mut next_version = Counter::new();
    let mut state = VarState::root();
    inline_vars_block(large, &mut state, &mut next_version, block);
}

fn inline_vars_block(large: &mut IrLargeArena, state: &mut VarState, next_version: &mut Counter, block: &mut IrBlock) {
    let IrBlock { statements } = block;

    for stmt in statements {
        match &mut stmt.inner {
            IrStatement::Assign(IrAssignmentTarget { base, steps }, source) => {
                // visit expressions
                for step in steps.iter_mut() {
                    match step {
                        IrTargetStep::ArrayIndex(index) => inline_vars_expr(large, state, index),
                        IrTargetStep::ArraySlice { start, len: _ } => inline_vars_expr(large, state, start),
                    }
                }
                inline_vars_expr(large, state, source);

                // record assignment
                if let IrSignalOrVariable::Variable(base) = *base {
                    let copy_of = if steps.is_empty()
                        && let IrExpression::Variable(source) = *source
                    {
                        let source_info = state.var_info(source);
                        Some((source, source_info.version))
                    } else {
                        None
                    };

                    let version = next_version.next();
                    state.map.insert(base, VarInfo { version, copy_of });
                }
            }
            IrStatement::Block(block) => {
                inline_vars_block(large, state, next_version, block);
            }
            IrStatement::If(IrIfStatement {
                condition,
                then_block,
                else_block,
            }) => {
                inline_vars_expr(large, state, condition);

                let mut state_then = state.child();
                inline_vars_block(large, &mut state_then, next_version, then_block);

                let mut state_else = state.child();
                if let Some(else_block) = else_block {
                    inline_vars_block(large, &mut state_else, next_version, else_block);
                }

                state.join_branches(next_version, state_then.map, state_else.map);
            }
            IrStatement::For(_) => {
                // clobber all variables that are written, and don't bother doing any replacements
                // TODO we could still do inlining for variables that are not written in the loop itself
                stmt.inner
                    .visit_values_accessed(large, &mut |var, access| match access {
                        ValueAccess::Read => {}
                        ValueAccess::Write => {
                            if let IrSignalOrVariable::Variable(var) = var {
                                let version = next_version.next();
                                state.map.insert(var, VarInfo { version, copy_of: None });
                            }
                        }
                    });
            }
            IrStatement::Print(pieces) => {
                for piece in pieces {
                    match piece {
                        IrStringPiece::Literal(_str) => {}
                        IrStringPiece::Substitute(sub) => match sub {
                            IrStringSubstitution::Integer(expr, _radix) => {
                                inline_vars_expr(large, state, expr);
                            }
                        },
                    }
                }
            }
        }
    }
}

fn inline_vars_expr(large: &mut IrLargeArena, state: &VarState, expr: &mut IrExpression) {
    let expr_new = expr.map_recursive(large, &mut |operand| {
        if let IrExpression::Variable(var) = *operand {
            let info = state.var_info(var);

            if let Some((other_var, other_version)) = info.copy_of {
                let other_info = state.var_info(other_var);
                if other_info.version == other_version {
                    return Some(IrExpression::Variable(other_var));
                }
            }
        }

        None
    });

    if let Some(new_expr) = expr_new {
        *expr = new_expr;
    }
}

impl VarState<'_> {
    fn root() -> Self {
        VarState {
            parent: None,
            map: Default::default(),
        }
    }

    fn child(&self) -> VarState<'_> {
        VarState {
            parent: Some(self),
            map: Default::default(),
        }
    }

    fn var_info(&self, var: IrVariable) -> VarInfo {
        let mut curr = self;
        loop {
            if let Some(&info) = curr.map.get(&var) {
                return info;
            }
            curr = match curr.parent {
                Some(parent) => parent,
                None => {
                    return VarInfo {
                        version: 0,
                        copy_of: None,
                    };
                }
            }
        }
    }

    fn join_branches(
        &mut self,
        next_version: &mut Counter,
        map_0: IndexMap<IrVariable, VarInfo>,
        map_1: IndexMap<IrVariable, VarInfo>,
    ) {
        let VarState { parent: _, map } = self;

        let changed_vars = chain!(map_0.keys(), map_1.keys().filter(|&k| !map_0.contains_key(k)));

        for &var in changed_vars {
            let version = next_version.next();
            map.insert(var, VarInfo { version, copy_of: None });
        }
    }
}

struct Counter(u64);

impl Counter {
    fn new() -> Self {
        Counter(1)
    }

    fn next(&mut self) -> u64 {
        let val = self.0;
        self.0 += 1;
        val
    }
}
