use crate::front::compile::{CompileItemContext, Port, Register, Variable, Wire};
use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::domain::ValueDomain;
use crate::front::types::HardwareType;
use crate::front::value::HardwareValue;
use crate::mid::ir::{IrAssignmentTargetBase, IrWires};
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;

// TODO expand to all possible values again?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Polarized<V> {
    pub inverted: bool,
    pub signal: V,
}

impl<V> Polarized<V> {
    pub fn new(signal: V) -> Self {
        Polarized {
            signal,
            inverted: false,
        }
    }

    pub fn invert(self) -> Self {
        Polarized {
            signal: self.signal,
            inverted: !self.inverted,
        }
    }

    pub fn map_inner<F, U>(self, f: F) -> Polarized<U>
    where
        F: FnOnce(V) -> U,
    {
        Polarized {
            signal: f(self.signal),
            inverted: self.inverted,
        }
    }

    pub fn try_map_inner<F, U, E>(self, f: F) -> Result<Polarized<U>, E>
    where
        F: FnOnce(V) -> Result<U, E>,
    {
        Ok(Polarized {
            signal: f(self.signal)?,
            inverted: self.inverted,
        })
    }
}

impl Polarized<Signal> {
    pub fn diagnostic_string(self, s: &CompileItemContext) -> String {
        let Polarized { inverted, signal } = self;
        let signal_str = signal.diagnostic_string(s);
        match inverted {
            false => signal_str.to_owned(),
            true => format!("(!{})", signal_str),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Signal {
    Port(Port),
    Wire(Wire),
    Register(Register),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum SignalOrVariable {
    Signal(Signal),
    Variable(Variable),
}

impl Signal {
    pub fn diagnostic_string<'c>(self, s: &'c CompileItemContext) -> &'c str {
        match self {
            Signal::Port(port) => &s.ports[port].name,
            Signal::Wire(wire) => s.wires[wire].diagnostic_str(),
            Signal::Register(reg) => s.registers[reg].id.diagnostic_str(),
        }
    }

    pub fn ty<'s>(
        self,
        ctx: &'s mut CompileItemContext,
        use_span: Span,
    ) -> Result<Spanned<&'s HardwareType>, ErrorGuaranteed> {
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].ty.as_ref()),
            Signal::Wire(wire) => ctx.wires[wire]
                .typed(ctx.refs, &ctx.wire_interfaces, use_span)
                .map(|typed| typed.ty),
            Signal::Register(reg) => Ok(ctx.registers[reg].ty.as_ref()),
        }
    }

    pub fn suggest_domain(
        self,
        ctx: &mut CompileItemContext,
        suggest_domain: Spanned<ValueDomain>,
    ) -> Result<Spanned<ValueDomain>, ErrorGuaranteed> {
        let diags = ctx.refs.diags;
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].domain.map_inner(ValueDomain::from_port_domain)),
            Signal::Wire(wire) => ctx.wires[wire].suggest_domain(&mut ctx.wire_interfaces, suggest_domain),
            Signal::Register(reg) => {
                let reg_info = &mut ctx.registers[reg];
                if let Some(domain) = reg_info.domain? {
                    Ok(domain.map_inner(ValueDomain::Sync))
                } else if let ValueDomain::Sync(suggest_domain_inner) = suggest_domain.inner {
                    let reg_domain = reg_info.suggest_domain(Spanned::new(suggest_domain.span, suggest_domain_inner));
                    Ok(reg_domain.map_inner(ValueDomain::Sync))
                } else {
                    Err(diags.report_internal_error(suggest_domain.span, "suggesting non-sync domain for register"))
                }
            }
        }
    }

    pub fn domain(self, ctx: &mut CompileItemContext, span: Span) -> Result<Spanned<ValueDomain>, ErrorGuaranteed> {
        let diags = ctx.refs.diags;
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].domain.map_inner(ValueDomain::from_port_domain)),
            Signal::Wire(wire) => ctx.wires[wire].domain(diags, &mut ctx.wire_interfaces, span),
            Signal::Register(reg) => ctx.registers[reg]
                .domain(diags, span)
                .map(|d| d.map_inner(ValueDomain::Sync)),
        }
    }

    pub fn suggest_ty<'s>(
        self,
        ctx: &'s mut CompileItemContext,
        ir_wires: &mut IrWires,
        suggest: Spanned<&HardwareType>,
    ) -> Result<Spanned<&'s HardwareType>, ErrorGuaranteed> {
        match self {
            Signal::Port(_) => self.ty(ctx, suggest.span),
            // TODO allow suggesting type for registers too?
            Signal::Register(_) => self.ty(ctx, suggest.span),
            Signal::Wire(wire) => ctx.wires[wire]
                .suggest_ty(ctx.refs, &ctx.wire_interfaces, ir_wires, suggest)
                .map(|typed| typed.ty),
        }
    }

    pub fn as_ir_target_base(
        self,
        ctx: &mut CompileItemContext,
        use_span: Span,
    ) -> Result<IrAssignmentTargetBase, ErrorGuaranteed> {
        match self {
            Signal::Port(port) => Ok(IrAssignmentTargetBase::Port(ctx.ports[port].ir)),
            Signal::Wire(wire) => Ok(IrAssignmentTargetBase::Wire(
                ctx.wires[wire].typed(ctx.refs, &ctx.wire_interfaces, use_span)?.ir,
            )),
            Signal::Register(reg) => Ok(IrAssignmentTargetBase::Register(ctx.registers[reg].ir)),
        }
    }

    pub fn as_hardware_value(self, ctx: &mut CompileItemContext, span: Span) -> Result<HardwareValue, ErrorGuaranteed> {
        let diags = ctx.refs.diags;
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].as_hardware_value()),
            Signal::Wire(wire) => ctx.wires[wire].as_hardware_value(ctx.refs, &mut ctx.wire_interfaces, span),
            Signal::Register(reg) => ctx.registers[reg].as_hardware_value(diags, span),
        }
    }
}
