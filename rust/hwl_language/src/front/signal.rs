use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::{PortDomain, ValueDomain};
use crate::front::flow::Variable;
use crate::front::item::{ElaboratedInterface, ElaboratedInterfaceView};
use crate::front::types::HardwareType;
use crate::front::value::HardwareValue;
use crate::mid::ir::{IrExpression, IrPort, IrSignal, IrWire, IrWireInfo, IrWires};
use crate::new_index_type;
use crate::syntax::ast::{DomainKind, Identifier, MaybeIdentifier, PortDirection};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::util::ResultExt;
use crate::util::arena::Arena;
use derive_more::From;

new_index_type!(pub Port);
new_index_type!(pub PortInterface);
new_index_type!(pub Wire);
new_index_type!(pub WireInterface);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Signal<P = Port, W = Wire> {
    Port(P),
    Wire(W),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, From)]
pub enum SignalOrVariable {
    Signal(Signal),
    Variable(Variable),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Polarized<V> {
    pub inverted: bool,
    pub signal: V,
}

// TODO move this stuff into signals?
#[derive(Debug)]
pub struct PortInfo {
    pub span: Span,
    pub name: String,
    pub direction: Spanned<PortDirection>,
    pub domain: Spanned<PortDomain<Port>>,
    pub ty: Spanned<HardwareType>,
    pub ir: IrPort,
    // TODO include interface this is a part of if any?
}

#[derive(Debug)]
pub struct PortInterfaceInfo {
    pub id: Identifier,
    pub view: Spanned<ElaboratedInterfaceView>,
    pub domain: Spanned<DomainKind<Polarized<Port>>>,
    pub ports: Vec<Port>,
}

#[derive(Debug)]
pub enum WireInfo {
    Single(WireInfoSingle),
    Interface(WireInfoInInterface),
}

#[derive(Debug)]
pub struct WireInfoSingle {
    pub id: MaybeIdentifier<Spanned<String>>,
    pub domain: DiagResult<Option<Spanned<ValueDomain>>>,
    pub typed: DiagResult<Option<WireInfoTyped<HardwareType>>>,
}

#[derive(Debug)]
pub struct WireInfoInInterface {
    pub decl_span: Span,
    pub interface: Spanned<WireInterface>,
    pub index: usize,
    pub diagnostic_string: String,
    pub ir: IrWire,
}

#[derive(Debug)]
pub struct WireInterfaceInfo {
    pub id: MaybeIdentifier<Spanned<String>>,
    pub domain: DiagResult<Option<Spanned<ValueDomain>>>,
    pub interface: Spanned<ElaboratedInterface>,
    pub wires: Vec<Wire>,
    // TODO rename
    pub ir_wires: Vec<IrWire>,
}

#[derive(Debug)]
pub struct WireInfoTyped<T> {
    pub ty: Spanned<T>,
    pub ir: IrWire,
}

impl<T> WireInfoTyped<T> {
    pub fn as_ref(&self) -> WireInfoTyped<&T> {
        WireInfoTyped {
            ty: self.ty.as_ref(),
            ir: self.ir,
        }
    }
}

impl PortInfo {
    pub fn as_hardware_value(&self) -> HardwareValue {
        HardwareValue {
            ty: self.ty.inner.clone(),
            domain: ValueDomain::from_port_domain(self.domain.inner),
            expr: IrExpression::Signal(IrSignal::Port(self.ir)),
        }
    }
}

impl WireInfo {
    pub fn decl_span(&self) -> Span {
        match self {
            WireInfo::Single(slf) => slf.id.span(),
            WireInfo::Interface(slf) => slf.decl_span,
        }
    }

    pub fn diagnostic_str(&self) -> &str {
        match self {
            WireInfo::Single(slf) => slf.id.diagnostic_str(),
            WireInfo::Interface(slf) => &slf.diagnostic_string,
        }
    }

    pub fn suggest_domain(
        &mut self,
        wire_interfaces: &mut Arena<WireInterface, WireInterfaceInfo>,
        suggest: Spanned<ValueDomain>,
    ) -> DiagResult<Spanned<ValueDomain>> {
        match self {
            WireInfo::Single(slf) => Ok(*slf.domain.as_ref_mut_ok()?.get_or_insert(suggest)),
            WireInfo::Interface(slf) => wire_interfaces[slf.interface.inner].suggest_domain(suggest),
        }
    }

    pub fn domain(
        &mut self,
        diags: &Diagnostics,
        wire_interfaces: &mut Arena<WireInterface, WireInterfaceInfo>,
        use_span: Span,
    ) -> DiagResult<Spanned<ValueDomain>> {
        let (decl_span, slot) = match self {
            WireInfo::Single(slf) => (slf.id.span(), &mut slf.domain),
            WireInfo::Interface(slf) => {
                let info = &mut wire_interfaces[slf.interface.inner];
                (info.id.span(), &mut info.domain)
            }
        };

        get_inferred(diags, "wire", "domain", slot, decl_span, use_span).copied()
    }

    pub fn suggest_ty<'s>(
        &'s mut self,
        refs: CompileRefs<'_, 's>,
        wire_interfaces: &Arena<WireInterface, WireInterfaceInfo>,
        ir_wires: &mut Arena<IrWire, IrWireInfo>,
        suggest: Spanned<&HardwareType>,
    ) -> DiagResult<WireInfoTyped<&'s HardwareType>> {
        match self {
            WireInfo::Single(slf) => {
                // take the suggestion into account
                Ok(slf
                    .typed
                    .as_ref_mut_ok()?
                    .get_or_insert_with(|| {
                        let ir = ir_wires.push(IrWireInfo {
                            ty: suggest.inner.as_ir(refs),
                            debug_info_id: slf.id.spanned_string(),
                            debug_info_ty: suggest.inner.clone().value_string(&refs.shared.elaboration_arenas),
                            // will be filled in later during the inference checking pass
                            debug_info_domain: String::new(),
                        });

                        WireInfoTyped {
                            ty: suggest.cloned(),
                            ir,
                        }
                    })
                    .as_ref())
            }
            WireInfo::Interface(slf) => {
                // ignore the suggestion, just get the type
                let wire_interface = &wire_interfaces[slf.interface.inner];
                let elab_interface = refs
                    .shared
                    .elaboration_arenas
                    .interface_info(wire_interface.interface.inner);

                Ok(WireInfoTyped {
                    ty: elab_interface.ports[slf.index].ty.as_ref_ok()?.as_ref(),
                    ir: slf.ir,
                })
            }
        }
    }

    pub fn expect_typed<'s>(
        &'s mut self,
        refs: CompileRefs<'_, 's>,
        wire_interfaces: &Arena<WireInterface, WireInterfaceInfo>,
        use_span: Span,
    ) -> DiagResult<WireInfoTyped<&'s HardwareType>> {
        match self {
            WireInfo::Single(slf) => {
                get_inferred(refs.diags, "wire", "type", &mut slf.typed, slf.id.span(), use_span).map(|ty| ty.as_ref())
            }
            WireInfo::Interface(slf) => {
                let wire_interface = &wire_interfaces[slf.interface.inner];
                let elab_interface = refs
                    .shared
                    .elaboration_arenas
                    .interface_info(wire_interface.interface.inner);

                Ok(WireInfoTyped {
                    ty: elab_interface.ports[slf.index].ty.as_ref_ok()?.as_ref(),
                    ir: slf.ir,
                })
            }
        }
    }

    pub fn typed_maybe<'s>(
        &'s mut self,
        refs: CompileRefs<'_, 's>,
        wire_interfaces: &Arena<WireInterface, WireInterfaceInfo>,
    ) -> DiagResult<Option<WireInfoTyped<&'s HardwareType>>> {
        match self {
            WireInfo::Single(slf) => slf
                .typed
                .as_ref_ok()
                .map(|typed| typed.as_ref().map(WireInfoTyped::as_ref)),
            WireInfo::Interface(slf) => {
                let wire_interface = &wire_interfaces[slf.interface.inner];
                let elab_interface = refs
                    .shared
                    .elaboration_arenas
                    .interface_info(wire_interface.interface.inner);

                Ok(Some(WireInfoTyped {
                    ty: elab_interface.ports[slf.index].ty.as_ref_ok()?.as_ref(),
                    ir: slf.ir,
                }))
            }
        }
    }

    pub fn as_hardware_value(
        &mut self,
        refs: CompileRefs,
        wire_interfaces: &mut Arena<WireInterface, WireInterfaceInfo>,
        use_span: Span,
    ) -> DiagResult<HardwareValue> {
        let domain = self.domain(refs.diags, wire_interfaces, use_span)?.inner;
        let typed = self.expect_typed(refs, wire_interfaces, use_span)?;

        Ok(HardwareValue {
            ty: typed.ty.inner.clone(),
            domain,
            expr: IrExpression::Signal(IrSignal::Wire(typed.ir)),
        })
    }
}

impl WireInterfaceInfo {
    pub fn suggest_domain(&mut self, suggest: Spanned<ValueDomain>) -> DiagResult<Spanned<ValueDomain>> {
        Ok(*self.domain.as_ref_mut_ok()?.get_or_insert(suggest))
    }
}

// TODO prevent cascading errors if a previous block has failed for some reason,
//   which caused certain things to not be inferred
fn get_inferred<'s, T>(
    diags: &Diagnostics,
    kind: &str,
    inferred: &str,
    slot: &'s mut DiagResult<Option<T>>,
    decl_span: Span,
    use_span: Span,
) -> DiagResult<&'s T> {
    match *slot {
        Ok(Some(ref inferred)) => Ok(inferred),
        Err(e) => Err(e),
        Ok(None) => {
            let e = DiagnosticError::new(
                format!("{kind} {inferred} is not yet known"),
                use_span,
                format!("{kind} used here before {inferred} could be inferred"),
            )
            .add_info(decl_span, format!("declared here without {inferred}"))
            .add_footer_hint(format!("explicitly add a {inferred} to the declaration"))
            .report(diags);
            *slot = Err(e);
            Err(e)
        }
    }
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
            true => format!("(!{signal_str})"),
        }
    }
}

impl Signal {
    pub fn diagnostic_string<'c>(self, s: &'c CompileItemContext) -> &'c str {
        match self {
            Signal::Port(port) => &s.ports[port].name,
            Signal::Wire(wire) => s.wires[wire].diagnostic_str(),
        }
    }

    pub fn suggest_domain(
        self,
        ctx: &mut CompileItemContext,
        suggest_domain: Spanned<ValueDomain>,
    ) -> DiagResult<Spanned<ValueDomain>> {
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].domain.map_inner(ValueDomain::from_port_domain)),
            Signal::Wire(wire) => ctx.wires[wire].suggest_domain(&mut ctx.wire_interfaces, suggest_domain),
        }
    }

    pub fn suggest_ty<'s>(
        self,
        ctx: &'s mut CompileItemContext,
        ir_wires: &mut IrWires,
        suggest: Spanned<&HardwareType>,
    ) -> DiagResult<Spanned<&'s HardwareType>> {
        match self {
            Signal::Port(_) => self.expect_ty(ctx, suggest.span),
            Signal::Wire(wire) => ctx.wires[wire]
                .suggest_ty(ctx.refs, &ctx.wire_interfaces, ir_wires, suggest)
                .map(|typed| typed.ty),
        }
    }

    pub fn domain(self, ctx: &mut CompileItemContext, span: Span) -> DiagResult<Spanned<ValueDomain>> {
        let diags = ctx.refs.diags;
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].domain.map_inner(ValueDomain::from_port_domain)),
            Signal::Wire(wire) => ctx.wires[wire].domain(diags, &mut ctx.wire_interfaces, span),
        }
    }

    // TODO rework this, there are a bunch of overlapping functions here
    pub fn expect_ty_and_ir<'s>(
        self,
        ctx: &'s mut CompileItemContext,
        use_span: Span,
    ) -> DiagResult<(Spanned<&'s HardwareType>, IrSignal)> {
        match self {
            Signal::Port(port) => {
                let info = &ctx.ports[port];
                Ok((info.ty.as_ref(), IrSignal::Port(info.ir)))
            }
            Signal::Wire(wire) => {
                let info = &ctx.wires[wire].expect_typed(ctx.refs, &ctx.wire_interfaces, use_span)?;
                Ok((info.ty, IrSignal::Wire(info.ir)))
            }
        }
    }

    pub fn expect_ty<'s>(
        self,
        ctx: &'s mut CompileItemContext,
        use_span: Span,
    ) -> DiagResult<Spanned<&'s HardwareType>> {
        self.expect_ty_and_ir(ctx, use_span).map(|(ty, _)| ty)
    }

    pub fn expect_ir(self, ctx: &mut CompileItemContext, use_span: Span) -> DiagResult<IrSignal> {
        self.expect_ty_and_ir(ctx, use_span).map(|(_, ir)| ir)
    }

    pub fn as_hardware_value(self, ctx: &mut CompileItemContext, span: Span) -> DiagResult<HardwareValue> {
        match self {
            Signal::Port(port) => Ok(ctx.ports[port].as_hardware_value()),
            Signal::Wire(wire) => ctx.wires[wire].as_hardware_value(ctx.refs, &mut ctx.wire_interfaces, span),
        }
    }
}

impl From<Port> for Signal {
    fn from(value: Port) -> Self {
        Signal::Port(value)
    }
}

impl From<Wire> for Signal {
    fn from(value: Wire) -> Self {
        Signal::Wire(value)
    }
}

impl From<Port> for SignalOrVariable {
    fn from(value: Port) -> Self {
        SignalOrVariable::Signal(Signal::Port(value))
    }
}

impl From<Wire> for SignalOrVariable {
    fn from(value: Wire) -> Self {
        SignalOrVariable::Signal(Signal::Wire(value))
    }
}
