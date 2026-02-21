use crate::front::block::{BlockEnd, join_block_ends_branches};
use crate::front::check::{TypeContainsReason, check_type_contains_value, check_type_is_range_compile};
use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, DiagnosticError, DiagnosticWarning};
use crate::front::exit::ExitStack;
use crate::front::flow::{Flow, FlowHardware, ImplicationContradiction, VariableId};
use crate::front::implication::{HardwareValueWithImplications, Implication, ValueWithImplications};
use crate::front::item::{ElaboratedEnum, ElaborationArenas, HardwareChecked};
use crate::front::scope::{NamedValue, Scope, ScopedEntry};
use crate::front::types::{HardwareType, NonHardwareType, Type, Typed};
use crate::front::value::{CompileCompoundValue, CompileValue, NotCompile, SimpleCompileValue, Value, ValueCommon};
use crate::mid::ir::{
    IrBlock, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIfStatement, IrIntCompareOp, IrLargeArena, IrStatement,
};
use crate::syntax::ast::{Block, BlockStatement, MatchBranch, MatchPattern, MatchStatement, MaybeIdentifier};
use crate::syntax::pos::{HasSpan, Pos, Span, Spanned};
use crate::util::big_int::BigInt;
use crate::util::range::Range;
use crate::util::range_multi::{AnyMultiRange, ClosedMultiRange, MultiRange};
use itertools::{Itertools, zip_eq};
use unwrap_match::unwrap_match;

#[derive(Debug)]
pub enum EvaluatedMatchPattern {
    Wildcard,
    WildcardVal(MaybeIdentifier),
    EqualTo(Spanned<CompileValue>),
    InRange(Spanned<Range<BigInt>>),
    IsEnumVariant {
        variant_index: usize,
        payload_id: Option<MaybeIdentifier>,
    },
}

#[derive(Debug)]
enum MatchCoverage {
    Bool {
        rem_false: bool,
        rem_true: bool,
    },
    Int {
        rem_range: ClosedMultiRange<BigInt>,
    },
    Enum {
        target_ty: HardwareChecked<ElaboratedEnum>,
        rem_variants: Vec<bool>,
    },
}

#[derive(Debug)]
enum CompileBranchMatched {
    Yes(Option<BranchDeclare<CompileValue>>),
    No,
}

#[derive(Debug)]
struct HardwareBranchMatched {
    cond: Option<IrExpression>,
    declare: Option<BranchDeclare<ValueWithImplications>>,
    implications: Vec<Implication>,
}

#[derive(Debug)]
pub struct BranchDeclare<V> {
    pattern_span: Span,
    id: MaybeIdentifier,
    value: V,
}

impl<V: Into<ValueWithImplications>> BranchDeclare<V> {
    pub fn declare(self, refs: CompileRefs, scope: &mut Scope, flow: &mut impl Flow) -> DiagResult<()> {
        let BranchDeclare {
            pattern_span,
            id,
            value,
        } = self;

        let var = flow.var_new_immutable_init(refs, id.span(), VariableId::Id(id), pattern_span, Ok(value.into()))?;
        scope.maybe_declare(
            refs.diags,
            Ok(id.spanned_str(refs.fixed.source)),
            Ok(ScopedEntry::Named(NamedValue::Variable(var))),
        );

        Ok(())
    }
}

impl MatchCoverage {
    fn as_diagnostic_string(&self, elab: &ElaborationArenas) -> String {
        match self {
            &MatchCoverage::Bool { rem_false, rem_true } => {
                let mut parts = vec![];
                if rem_false {
                    parts.push(false);
                }
                if rem_true {
                    parts.push(true);
                }
                format!("{parts:?}")
            }
            MatchCoverage::Int { rem_range } => rem_range.to_string(),
            MatchCoverage::Enum {
                target_ty,
                rem_variants,
            } => {
                let parts = rem_variants
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &x)| {
                        if x {
                            Some(format!(
                                ".{}",
                                elab.enum_info(target_ty.inner()).variants[i].debug_info_name
                            ))
                        } else {
                            None
                        }
                    })
                    .join(", ");
                format!("[{}]", parts)
            }
        }
    }

    fn any(&self) -> bool {
        match self {
            &MatchCoverage::Bool { rem_false, rem_true } => rem_false || rem_true,
            MatchCoverage::Int { rem_range } => !rem_range.is_empty(),
            MatchCoverage::Enum {
                target_ty: _,
                rem_variants,
            } => rem_variants.iter().any(|&x| x),
        }
    }

    fn clear(&mut self) {
        match self {
            MatchCoverage::Bool { rem_false, rem_true } => {
                *rem_false = false;
                *rem_true = false;
            }
            MatchCoverage::Int { rem_range } => *rem_range = ClosedMultiRange::EMPTY,
            MatchCoverage::Enum {
                target_ty: _,
                rem_variants,
            } => {
                rem_variants.fill(false);
            }
        }
    }
}

impl CompileItemContext<'_, '_> {
    pub fn elaborate_match_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt: &MatchStatement<Block<BlockStatement>>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let &MatchStatement {
            span_keyword,
            target,
            pos_end,
            ref branches,
        } = stmt;

        // eval target
        let target = self.eval_expression_with_implications(scope, flow, &Type::Any, target)?;
        let target_ty = target.inner.ty();

        // eval branches
        let branches_evaluated =
            self.elaborate_match_statement_eval_branches(scope, flow, target.span, &target_ty, branches)?;

        // dispatch on target
        match CompileValue::try_from(&target.inner) {
            Ok(target_inner) => {
                // compile-time target value, we can handle the entire match at compile-time
                let target = Spanned::new(target.span, target_inner);
                self.elaborate_match_statement_compile(scope, flow, stack, target, pos_end, branches_evaluated)
            }
            Err(NotCompile) => {
                // hardware target value (at least partially), convert the target to full hardware
                //   and handle the match at hardware time
                let flow = flow.require_hardware(span_keyword, "match on hardware target")?;

                let target_ty = target_ty.as_hardware_type(elab).map_err(|_: NonHardwareType| {
                    diags.report_error_simple(
                        "failed to fully convert non-compile match target to hardware",
                        target.span,
                        format!("match target has non-hardware type `{}`", target_ty.value_string(elab)),
                    )
                })?;

                let target_inner = match target.inner {
                    ValueWithImplications::Simple(t) => HardwareValueWithImplications::simple(
                        t.as_hardware_value_unchecked(self.refs, &mut self.large, target.span, target_ty.clone())?,
                    ),
                    ValueWithImplications::Compound(t) => HardwareValueWithImplications::simple(
                        t.as_hardware_value_unchecked(self.refs, &mut self.large, target.span, target_ty.clone())?,
                    ),
                    ValueWithImplications::Hardware(t) => t,
                };
                let target = Spanned::new(target.span, target_inner);

                self.elaborate_match_statement_hardware(
                    scope,
                    flow,
                    stack,
                    span_keyword,
                    target,
                    pos_end,
                    branches_evaluated,
                )
            }
        }
    }

    pub fn compile_match_statement_choose_branch<'a, B>(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stmt: &'a MatchStatement<B>,
    ) -> DiagResult<(Option<BranchDeclare<CompileValue>>, &'a B)> {
        let &MatchStatement {
            span_keyword,
            target,
            pos_end,
            ref branches,
        } = stmt;

        // eval target
        let reason = Spanned::new(span_keyword, "compile-time match");
        let target = self.eval_expression_as_compile(scope, flow, &Type::Any, target, reason)?;
        let target_ty = target.inner.ty();

        // eval branches
        let branches_evaluated =
            self.elaborate_match_statement_eval_branches(scope, flow, target.span, &target_ty, branches)?;

        // choose branch
        self.elaborate_match_statement_compile_choose_branch(target, pos_end, branches_evaluated)
    }

    pub fn elaborate_match_statement_eval_branches<'a, B>(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        target_span: Span,
        target_ty: &Type,
        branches: &'a Vec<MatchBranch<B>>,
    ) -> DiagResult<Vec<(Spanned<EvaluatedMatchPattern>, &'a B)>> {
        let mut result = vec![];
        for branch in branches {
            let MatchBranch { pattern, block } = branch;
            let pattern_eval = self.eval_match_pattern(scope, flow, target_span, target_ty, pattern.as_ref())?;
            result.push((Spanned::new(pattern.span, pattern_eval), block));
        }
        Ok(result)
    }

    fn eval_match_pattern(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        target_span: Span,
        target_ty: &Type,
        pattern: Spanned<&MatchPattern>,
    ) -> DiagResult<EvaluatedMatchPattern> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        match *pattern.inner {
            MatchPattern::Wildcard => Ok(EvaluatedMatchPattern::Wildcard),
            MatchPattern::WildcardVal(id) => Ok(EvaluatedMatchPattern::WildcardVal(id)),
            MatchPattern::EqualTo(value) => {
                let value = self.eval_expression_as_compile(
                    scope,
                    flow,
                    target_ty,
                    value,
                    Spanned::new(pattern.span, "match branch"),
                )?;
                Ok(EvaluatedMatchPattern::EqualTo(value))
            }
            MatchPattern::InRange { span_in, range } => {
                let value = self.eval_expression_as_compile(
                    scope,
                    flow,
                    target_ty,
                    range,
                    Spanned::new(pattern.span, "match branch"),
                )?;
                let value = check_type_is_range_compile(diags, elab, TypeContainsReason::Operator(span_in), value)?;
                Ok(EvaluatedMatchPattern::InRange(Spanned::new(range.span, value)))
            }
            MatchPattern::IsEnumVariant { variant, payload_id } => {
                if let &Type::Enum(target_ty) = target_ty {
                    let enum_info = self.refs.shared.elaboration_arenas.enum_info(target_ty);

                    let variant_str = variant.spanned_str(self.refs.fixed.source);
                    let variant_index = enum_info.find_variant(diags, variant_str)?;
                    let variant_info = &enum_info.variants[variant_index];
                    let variant_name = &variant_info.debug_info_name;

                    match (payload_id, &variant_info.payload_ty) {
                        (None, None) | (Some(_), Some(_)) => {}
                        (None, Some(variant_pyload_ty)) => {
                            let hint = format!(
                                "add a payload to the pattern: `.{variant_name}(_)` or `.{variant_name}(val payload)`"
                            );
                            let diag = DiagnosticError::new(
                                "enum pattern payload mismatch",
                                pattern.span,
                                "pattern without a payload here",
                            )
                            .add_info(variant_pyload_ty.span, "variant declared with a payload here")
                            .add_footer_hint(hint)
                            .report(diags);
                            return Err(diag);
                        }
                        (Some(payload), None) => {
                            let diag = DiagnosticError::new(
                                "enum pattern payload mismatch",
                                payload.span(),
                                "pattern with a payload here",
                            )
                            .add_info(variant_info.id.span, "variant declared without a payload here")
                            .add_footer_hint(format!("remove the payload from the pattern: `.{variant_name}`"))
                            .report(diags);
                            return Err(diag);
                        }
                    }

                    Ok(EvaluatedMatchPattern::IsEnumVariant {
                        variant_index,
                        payload_id,
                    })
                } else {
                    Err(DiagnosticError::new(
                        "enum match pattern with non-enum target type",
                        pattern.span,
                        "enum match pattern used here",
                    )
                    .add_info(
                        target_span,
                        format!("target has type `{}`", target_ty.value_string(elab)),
                    )
                    .report(diags))
                }
            }
        }
    }

    fn elaborate_match_statement_compile(
        &mut self,
        scope_parent: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        target: Spanned<CompileValue>,
        pos_end: Pos,
        branches: Vec<(Spanned<EvaluatedMatchPattern>, &Block<BlockStatement>)>,
    ) -> DiagResult<BlockEnd> {
        let (declare, block) = self.elaborate_match_statement_compile_choose_branch(target, pos_end, branches)?;

        let mut scope_branch = scope_parent.new_child(block.span);
        if let Some(declare) = declare {
            declare.declare(self.refs, &mut scope_branch, flow)?;
        }

        self.elaborate_block(&scope_branch, flow, stack, block)
    }

    fn elaborate_match_statement_compile_choose_branch<'a, B>(
        &mut self,
        target: Spanned<CompileValue>,
        pos_end: Pos,
        branches: Vec<(Spanned<EvaluatedMatchPattern>, &'a B)>,
    ) -> DiagResult<(Option<BranchDeclare<CompileValue>>, &'a B)> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        // compile-time match, just check each pattern in sequence with early exit
        for (pattern, branch) in branches {
            let matched = match pattern.inner {
                EvaluatedMatchPattern::Wildcard => CompileBranchMatched::Yes(None),
                EvaluatedMatchPattern::WildcardVal(id) => CompileBranchMatched::Yes(Some(BranchDeclare {
                    pattern_span: pattern.span,
                    id,
                    value: target.inner.clone(),
                })),
                EvaluatedMatchPattern::EqualTo(value) => {
                    if target.inner == value.inner {
                        CompileBranchMatched::Yes(None)
                    } else {
                        CompileBranchMatched::No
                    }
                }
                EvaluatedMatchPattern::InRange(range) => {
                    if let CompileValue::Simple(SimpleCompileValue::Int(target)) = &target.inner
                        && range.inner.contains(target)
                    {
                        CompileBranchMatched::Yes(None)
                    } else {
                        CompileBranchMatched::No
                    }
                }
                EvaluatedMatchPattern::IsEnumVariant {
                    variant_index,
                    payload_id,
                } => {
                    if let CompileValue::Compound(CompileCompoundValue::Enum(target)) = &target.inner
                        && target.variant == variant_index
                    {
                        let declare = match (payload_id, &target.payload) {
                            (None, None) => None,
                            (Some(payload_id), Some(target_payload)) => Some(BranchDeclare {
                                pattern_span: pattern.span,
                                id: payload_id,
                                value: target_payload.as_ref().clone(),
                            }),
                            (None, Some(_)) | (Some(_), None) => {
                                return Err(diags.report_error_internal(pattern.span, "payload mismatch"));
                            }
                        };

                        CompileBranchMatched::Yes(declare)
                    } else {
                        CompileBranchMatched::No
                    }
                }
            };

            match matched {
                CompileBranchMatched::Yes(declare) => {
                    return Ok((declare, branch));
                }
                CompileBranchMatched::No => {
                    // continue to next branch
                }
            }
        }

        Err(DiagnosticError::new(
            "compile-time match statement reached end without matching any branch",
            Span::empty_at(pos_end),
            "did not match any branch",
        )
        .add_info(
            target.span,
            format!("target value `{}`", target.inner.value_string(elab)),
        )
        .report(diags))
    }

    fn elaborate_match_statement_hardware(
        &mut self,
        scope_parent: &Scope,
        flow_parent: &mut FlowHardware,
        stack: &mut ExitStack,
        span_keyword: Span,
        target: Spanned<HardwareValueWithImplications>,
        pos_end: Pos,
        branches: Vec<(Spanned<EvaluatedMatchPattern>, &Block<BlockStatement>)>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let target_version = target.inner.version;
        let target_value = target.inner;
        let target_domain = Spanned::new(target.span, target_value.value.domain);

        // TODO extract function
        let mut coverage_remaining = match &target_value.value.ty {
            HardwareType::Bool => MatchCoverage::Bool {
                rem_false: true,
                rem_true: true,
            },
            HardwareType::Int(ty) => MatchCoverage::Int {
                rem_range: ClosedMultiRange::from(ty.clone()),
            },
            &HardwareType::Enum(ty) => {
                let elab_enum = elab.enum_info(ty.inner());
                MatchCoverage::Enum {
                    target_ty: ty,
                    rem_variants: vec![true; elab_enum.variants.len()],
                }
            }
            _ => {
                return Err(diags.report_error_todo(
                    target.span,
                    format!(
                        "hardware matching for target type `{}`",
                        target_value.value.ty.value_string(elab)
                    ),
                ));
            }
        };

        let mut all_conditions = vec![];
        let mut all_contents = vec![];
        let mut all_ends = vec![];

        let warn_unreachable_branch =
            |span: Span, coverage_remaining: &MatchCoverage, branch_pattern: Option<String>| {
                let message_branch = if let Some(branch_pattern) = branch_pattern {
                    format!("this branch with pattern `{branch_pattern}` can never match")
                } else {
                    "this branch can never match".to_owned()
                };

                let message_footer = if coverage_remaining.any() {
                    format!(
                        "the remaining uncovered patterns are: `{}`",
                        coverage_remaining.as_diagnostic_string(elab)
                    )
                } else {
                    "all possible patterns are already covered".to_owned()
                };

                DiagnosticWarning::new("unreachable match branch", span, message_branch)
                    .add_info(
                        target.span,
                        format!("target type `{}`", target_value.value.ty.value_string(elab)),
                    )
                    .add_footer_info(message_footer)
                    .report(diags);
            };

        for (branch_pattern, branch_block) in branches {
            let matched = match branch_pattern.inner {
                EvaluatedMatchPattern::Wildcard => {
                    if !coverage_remaining.any() {
                        warn_unreachable_branch(branch_pattern.span, &coverage_remaining, None);
                        continue;
                    }
                    coverage_remaining.clear();

                    HardwareBranchMatched {
                        cond: None,
                        declare: None,
                        implications: vec![],
                    }
                }
                EvaluatedMatchPattern::WildcardVal(id) => {
                    if !coverage_remaining.any() {
                        warn_unreachable_branch(branch_pattern.span, &coverage_remaining, None);
                        continue;
                    }
                    coverage_remaining.clear();

                    HardwareBranchMatched {
                        cond: None,
                        declare: Some(BranchDeclare {
                            pattern_span: branch_pattern.span,
                            id,
                            value: Value::Hardware(target_value.clone()),
                        }),
                        implications: vec![],
                    }
                }
                EvaluatedMatchPattern::EqualTo(value) => {
                    check_type_contains_value(
                        diags,
                        elab,
                        TypeContainsReason::MatchPattern(value.span),
                        &target_value.value.ty.as_type(),
                        value.as_ref(),
                    )?;

                    let (cond, implications) = match &mut coverage_remaining {
                        MatchCoverage::Bool { rem_false, rem_true } => {
                            let value = unwrap_match!(value.inner, CompileValue::Simple(SimpleCompileValue::Bool(value)) => value);

                            // TODO should we also imply that the bool itself is true/false? How does this work for ifs?
                            //  Or do the implications already cover that?
                            if value {
                                if !*rem_true {
                                    warn_unreachable_branch(
                                        branch_pattern.span,
                                        &coverage_remaining,
                                        Some("true".to_owned()),
                                    );
                                    continue;
                                }
                                *rem_true = false;

                                (
                                    target_value.value.expr.clone(),
                                    target_value.implications.if_true.clone(),
                                )
                            } else {
                                if !*rem_false {
                                    warn_unreachable_branch(
                                        branch_pattern.span,
                                        &coverage_remaining,
                                        Some("false".to_owned()),
                                    );
                                    continue;
                                }
                                *rem_false = false;

                                let cond = self
                                    .large
                                    .push_expr(IrExpressionLarge::BoolNot(target_value.value.expr.clone()));
                                (cond, target_value.implications.if_false.clone())
                            }
                        }
                        MatchCoverage::Int { rem_range } => {
                            let value = unwrap_match!(value.inner, CompileValue::Simple(SimpleCompileValue::Int(value)) => value);

                            if !rem_range.contains(&value) {
                                warn_unreachable_branch(
                                    branch_pattern.span,
                                    &coverage_remaining,
                                    Some(value.to_string()),
                                );
                                continue;
                            }
                            let value_range = MultiRange::from(Range::single(value.clone()));
                            *rem_range = rem_range.subtract(&value_range);

                            let cond = self.large.push_expr(IrExpressionLarge::IntCompare(
                                IrIntCompareOp::Eq,
                                target_value.value.expr.clone(),
                                IrExpression::Int(value.clone()),
                            ));

                            let implications = if let Some(target_version) = target_version {
                                vec![Implication::new_int(target_version, value_range)]
                            } else {
                                vec![]
                            };
                            (cond, implications)
                        }
                        MatchCoverage::Enum { .. } => {
                            return Err(diags.report_error_todo(branch_pattern.span, "matching enum value by equality"));
                        }
                    };

                    HardwareBranchMatched {
                        cond: Some(cond),
                        declare: None,
                        implications,
                    }
                }
                EvaluatedMatchPattern::InRange(range) => match &mut coverage_remaining {
                    MatchCoverage::Int { rem_range } => {
                        // TODO warn if parts of range already covered
                        // TODO share code with ordinary comparisons?
                        let range_multi = MultiRange::from(range.inner.clone());

                        if rem_range.intersect(&range_multi).is_empty() {
                            warn_unreachable_branch(
                                branch_pattern.span,
                                &coverage_remaining,
                                Some(format!("in {}", range.inner)),
                            );
                            continue;
                        }
                        *rem_range = rem_range.subtract(&range_multi);

                        let cond =
                            build_ir_int_in_range(&mut self.large, &target_value.value.expr, range.inner.clone());

                        let implications = if let Some(target_version) = target_version {
                            vec![Implication::new_int(target_version, range_multi)]
                        } else {
                            vec![]
                        };

                        HardwareBranchMatched {
                            cond,
                            declare: None,
                            implications,
                        }
                    }
                    _ => {
                        let diag = DiagnosticError::new(
                            "range match pattern with non-integer target type",
                            range.span,
                            "integer range match pattern used here",
                        )
                        .add_info(
                            target.span,
                            format!("target has type `{}`", target_value.value.ty.value_string(elab)),
                        )
                        .report(diags);
                        return Err(diag);
                    }
                },
                EvaluatedMatchPattern::IsEnumVariant {
                    variant_index,
                    payload_id,
                } => match &mut coverage_remaining {
                    MatchCoverage::Enum {
                        target_ty,
                        rem_variants,
                    } => {
                        if !rem_variants[variant_index] {
                            let branch_pattern_string = format!(
                                ".{}",
                                elab.enum_info(target_ty.inner()).variants[variant_index].debug_info_name
                            );
                            warn_unreachable_branch(
                                branch_pattern.span,
                                &coverage_remaining,
                                Some(branch_pattern_string),
                            );
                            continue;
                        }
                        rem_variants[variant_index] = false;

                        let enum_info = elab.enum_info(target_ty.inner());
                        let enum_info_hw = enum_info.hw.as_ref().unwrap();

                        let cond = enum_info_hw.check_tag_matches(
                            &mut self.large,
                            target_value.value.expr.clone(),
                            variant_index,
                        );
                        let payload_hw =
                            enum_info_hw.extract_payload(&mut self.large, &target_value.value, variant_index);
                        let declare = match (payload_id, payload_hw) {
                            (None, None) => None,
                            (Some(payload_id), Some(payload_value)) => Some(BranchDeclare {
                                pattern_span: branch_pattern.span,
                                id: payload_id,
                                value: Value::Hardware(HardwareValueWithImplications::simple(payload_value)),
                            }),
                            (None, Some(_)) | (Some(_), None) => {
                                return Err(diags.report_error_internal(branch_pattern.span, "payload mismatch"));
                            }
                        };

                        HardwareBranchMatched {
                            cond: Some(cond),
                            declare,
                            implications: vec![],
                        }
                    }
                    _ => {
                        let diag = DiagnosticError::new(
                            "enum variant match pattern with non-enum target type",
                            branch_pattern.span,
                            "enum variant match pattern used here",
                        )
                        .add_info(
                            target.span,
                            format!("target has type `{}`", target_value.value.ty.value_string(elab)),
                        )
                        .report(diags);
                        return Err(diag);
                    }
                },
            };

            let HardwareBranchMatched {
                cond,
                declare,
                implications,
            } = matched;

            let branch_span = branch_pattern.span.join(branch_block.span);
            let mut branch_scope = scope_parent.new_child(branch_span);
            let mut branch_flow = match flow_parent.new_child_branch(self, branch_span, target_domain, implications)? {
                Ok(branch_flow) => branch_flow,
                Err(ImplicationContradiction) => continue,
            };

            if let Some(declare) = declare {
                declare.declare(self.refs, &mut branch_scope, &mut branch_flow.as_flow())?;
            }

            let branch_end = self.elaborate_block(&branch_scope, &mut branch_flow.as_flow(), stack, branch_block)?;
            let content = branch_flow.finish();

            all_conditions.push(cond);
            all_contents.push(content);
            all_ends.push(branch_end);
        }

        // check coverage
        if coverage_remaining.any() {
            let mut diag = DiagnosticError::new(
                "hardware match statement must be exhaustive",
                Span::empty_at(pos_end),
                format!(
                    "patterns not covered: `{}`",
                    coverage_remaining.as_diagnostic_string(elab)
                ),
            )
            .add_info(
                target.span,
                format!("target type `{}`", target_value.value.ty.value_string(elab)),
            );

            if let MatchCoverage::Enum { target_ty, .. } = &coverage_remaining {
                let enum_info = elab.enum_info(target_ty.inner());
                diag = diag.add_info(enum_info.unique.id().span(), "enum declared here");
            }

            let diag = diag.add_footer_hint(
                "either add extra branches, or add a wildcard branch to ignore all remaining patterns: `_ => {}`",
            );

            return Err(diag.report(diags));
        }

        // join things
        let all_blocks = flow_parent.join_child_branches(self.refs, &mut self.large, span_keyword, all_contents)?;
        let joined_end = join_block_ends_branches(&all_ends);

        // build the if statement
        // TODO flatten this into single if/else-if/else structure or even a match?
        let mut joined_statement = None;
        for (cond, block) in zip_eq(all_conditions.into_iter().rev(), all_blocks.into_iter().rev()) {
            joined_statement = match cond {
                None => Some(IrStatement::Block(block)),
                Some(cond) => Some(IrStatement::If(IrIfStatement {
                    condition: cond,
                    then_block: block,
                    else_block: joined_statement.map(|curr| IrBlock::new_single(span_keyword, curr)),
                })),
            };
        }

        if let Some(curr) = joined_statement {
            flow_parent.push_ir_statement(Spanned::new(span_keyword, curr));
        }

        Ok(joined_end)
    }
}

fn build_ir_int_in_range(large: &mut IrLargeArena, value: &IrExpression, range: Range<BigInt>) -> Option<IrExpression> {
    range.assert_valid();
    let Range { start, end } = range;

    let cond_start = start.map(|start| {
        large.push_expr(IrExpressionLarge::IntCompare(
            IrIntCompareOp::Lte,
            IrExpression::Int(start),
            value.clone(),
        ))
    });
    let cond_end = end.map(|end| {
        large.push_expr(IrExpressionLarge::IntCompare(
            IrIntCompareOp::Lt,
            value.clone(),
            IrExpression::Int(end),
        ))
    });

    match (cond_start, cond_end) {
        (None, None) => None,
        (Some(cond), None) | (None, Some(cond)) => Some(cond),
        (Some(cond_start), Some(cond_end)) => {
            Some(large.push_expr(IrExpressionLarge::BoolBinary(IrBoolBinaryOp::And, cond_start, cond_end)))
        }
    }
}
