from pathlib import Path

from khdl_sandbox.common.util import compile_custom, diag_error

INTF_FOO = """
interface Foo {
    value: uint(4),
    ready: bool,
    view In { value: in, ready: out }
    view Out { value: out, ready: in }
}
"""

INTF_BUS = """
interface Bus { value: uint(4), view In { value: in } view Out { value: out } }
"""


def test_connect_port_port(tmp_dir: Path):
    src = INTF_FOO + """
    module top ports(x: interface async Foo.In, y: interface async Foo.Out) {
        connect(x view In, y view Out);
    }
    """
    inst = compile_custom(src).resolve_module("top.top").as_verilated(tmp_dir).instance()
    for v in range(4):
        inst.ports.x_value.value = v
        inst.ports.y_ready.value = v % 2 == 1
        inst.step(1)
        assert inst.ports.y_value.value == v
        assert inst.ports.x_ready.value == (v % 2 == 1)


def test_connect_through_wire(tmp_dir: Path):
    src = INTF_FOO + """
    module top ports(x: interface async Foo.In, y: interface async Foo.Out) {
        wire w: interface Foo;
        connect(x, w view Out);
        connect(w view In, y view Out);
    }
    """
    inst = compile_custom(src).resolve_module("top.top").as_verilated(tmp_dir).instance()
    for v in range(4):
        inst.ports.x_value.value = v
        inst.ports.y_ready.value = v % 2 == 0
        inst.step(1)
        assert inst.ports.y_value.value == v
        assert inst.ports.x_ready.value == (v % 2 == 0)


def test_connect_wire_with_instance(tmp_dir: Path):
    src = INTF_BUS + """
    module pass ports(a: interface async Bus.In, b: interface async Bus.Out) {
        comb { b.value = (a.value + 1) % 16; }
    }
    module top ports(x: interface async Bus.In, y: interface async Bus.Out) {
        wire v: interface Bus;
        wire w: interface Bus;
        connect(x view In, v view Out);
        instance pass ports(a=v, b=w);
        connect(w view In, y view Out);
    }
    """
    inst = compile_custom(src).resolve_module("top.top").as_verilated(tmp_dir).instance()
    for v in range(4):
        inst.ports.x_value.value = v
        inst.step(1)
        assert inst.ports.y_value.value == v + 1


def test_connect_array(tmp_dir: Path):
    src = INTF_BUS + """
    module top ports(
        async {
            x: interface [2][3]Bus.In,
            y: interface [2][3]Bus.Out,
            z: interface [6]Bus.In,
            u: interface [4]Bus.Out,
        }
    ) {
        wire w: interface [2][3]Bus;
        connect(x view In, w view Out);
        connect(w view In, y view Out);

        connect(z[2..6] view In, u view Out);
    }
    """
    inst = compile_custom(src).resolve_module("top.top").as_verilated(tmp_dir).instance()
    for i in range(2):
        for j in range(3):
            inst.ports[f"x_{i}_{j}_value"].value = 3 * i + j
    for i in range(6):
        inst.ports[f"z_{i}_value"].value = i + 1
    inst.step(1)
    for i in range(2):
        for j in range(3):
            assert inst.ports[f"y_{i}_{j}_value"].value == 3 * i + j
    for i in range(4):
        assert inst.ports[f"u_{i}_value"].value == i + 3


def test_connect_fanout(tmp_dir: Path):
    src = INTF_BUS + """
    module top ports(
        async {
            x: interface Bus.In,
            y: interface [3]Bus.Out,
            z: interface Bus.Out,
        }
    ) {
        wire w: interface Bus;
        connect(
            x view In,
            w view Out,
            for (i in 0..3) {
                y[i] view Out,
            }
        );
        connect(w view In, z view Out);
    }
    """
    inst = compile_custom(src).resolve_module("top.top").as_verilated(tmp_dir).instance()
    for v in range(4):
        inst.ports.x_value.value = v
        inst.step(1)
        for i in range(3):
            assert inst.ports[f"y_{i}_value"].value == v
        assert inst.ports.z_value.value == v


def test_connect_through_wire_reversed():
    src = INTF_FOO + """
    module top ports(x: interface async Foo.In, y: interface async Foo.Out) {
        wire w: interface Foo;
        connect(w view In, y view Out);
        connect(x view In, w view Out);
    }
    """
    _ = compile_custom(src).resolve_module("top.top")


def test_connect_wire_domain_inferred():
    src = INTF_BUS + """
    module sink ports(clk: in clock, a: interface sync(clk) Bus.In) {}
    module top ports(clk_a: in clock, clk_b: in clock, x: interface sync(clk_a) Bus.In) {
        wire w: interface Bus;
        connect(x view In, w view Out);
        instance sink ports(clk=clk_b, a=w);
    }
    """
    with diag_error("invalid domain crossing: different clock"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_domain_mismatch():
    src = INTF_BUS + """
    module top ports(clk: in clock, x: interface async Bus.In, y: interface sync(clk) Bus.Out) {
        connect(x view In, y view Out);
    }
    """
    with diag_error("invalid domain crossing: async to sync"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_wires_unknown_domain():
    src = INTF_BUS + """
    module top ports() {
        wire v: interface Bus;
        wire w: interface Bus;
        connect(v view In, w view Out);
    }
    """
    with diag_error("cannot infer domain for connection between wires"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_wire_driver_unknown_domain():
    src = INTF_BUS + """
    module top ports(clk: in clock, y: interface sync(clk) Bus.Out) {
        wire w: interface Bus;
        connect(w view In, y view Out);
    }
    """
    with diag_error("cannot infer domain for connection between wires"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_no_source():
    src = INTF_BUS + """
    module top ports(async { x: interface Bus.Out, y: interface Bus.Out }) {
        connect(x view Out, y view Out);
    }
    """
    with diag_error("signal `value` has no driver"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_multiple_sources():
    src = INTF_BUS + """
    module top ports(async { x: interface Bus.In, y: interface Bus.In }) {
        connect(x view In, y view In);
    }
    """
    with diag_error("signal `value` has multiple drivers"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_port_view_mismatch():
    src = INTF_BUS + """
    module top ports(async { x: interface Bus.In, y: interface Bus.Out }) {
        connect(x view Out, y view Out);
    }
    """
    with diag_error("view `Out` does not match the directions of the port interface"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_interface_mismatch():
    src = INTF_BUS + """
    interface Other { value: uint(4), view In { value: in } view Out { value: out } }
    module top ports(async { x: interface Bus.In, y: interface Other.Out }) {
        connect(x view In, y view Out);
    }
    """
    with diag_error("interface mismatch"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_shape_mismatch():
    src = INTF_BUS + """
    module top ports(async { x: interface [3]Bus.In, z: interface [1]Bus.Out }) {
        connect(x[0] view In, z view Out);
    }
    """
    with diag_error("interface array shape mismatch"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_single_entry():
    src = INTF_BUS + """
    module top ports(async { x: interface Bus.In }) {
        connect(x view In);
    }
    """
    with diag_error("connect requires at least two entries"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_non_interface():
    src = INTF_BUS + """
    module top ports(async { x: in uint(4), y: interface Bus.Out }) {
        connect(x view In, y view Out);
    }
    """
    with diag_error("expected interface value"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_multiple_drivers():
    src = INTF_BUS + """
    module top ports(async { x: interface Bus.In, y: interface Bus.Out }) {
        connect(x view In, y view Out);
        comb { y.value = 0; }
    }
    """
    with diag_error("port `y_value` has multiple drivers"):
        _ = compile_custom(src).resolve_module("top.top")


def test_connect_wire_missing_view():
    src = INTF_BUS + """
    module top ports(async { x: interface Bus.In }) {
        wire w: interface Bus;
        connect(x, w);
    }
    """
    with diag_error("missing view for wire interface"):
        _ = compile_custom(src).resolve_module("top.top")