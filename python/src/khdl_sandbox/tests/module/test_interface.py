from pathlib import Path

import khdl

from khdl_sandbox.common.util import compile_custom, diag_error


def test_interface_access(tmp_dir: Path):
    src = """
    interface Foo {
        x: bool,
        view input { x: in }
        view output { x: out }
    }
    module top ports(a: interface async Foo.input, b: interface async Foo.output) {
        comb {
            b.x = a.x;
        }   
    }
    """
    c = compile_custom(src)
    top: khdl.Module = c.resolve("top.top")
    print(top.as_verilog().source)
    top_verilated: khdl.ModuleVerilated = top.as_verilated(tmp_dir)
    top_inst: khdl.VerilatedInstance = top_verilated.instance()

    for v in [False, True, False, True]:
        top_inst.ports.a.x.value = v
        top_inst.step(1)
        assert top_inst.ports.b.x.value == v


def test_interface_chain(tmp_dir: Path):
    src = """
    interface foo {
        d: uint(8),
        view input { d: in }
        view output { d: out }
    }
    module top ports(x: interface async foo.input, y: interface async foo.output) {
        wire w: interface foo;
        instance pass ports(x=x, y=w);  
        instance pass ports(x=w, y=y);  
    }
    module pass ports(x: interface async foo.input, y: interface async foo.output) {
        comb { y.d = x.d; } 
    }
    """
    m: khdl.Module = compile_custom(src).resolve("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for v in [0, 1, 2, 3]:
        inst.ports.x.d.value = v
        inst.step(1)
        assert inst.ports.y.d.value == v


def test_interface_array_1d_index(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(
        async {
            x: interface [4]Foo.In,
            y: out uint(6),
        }
    ) {
        comb {
            var r = 0;
            for (i in 0..4) { r += x[i].value; }
            y = r;
        }
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(4):
        inst.ports.x[i].value.value = i + 1
    inst.step(1)
    assert inst.ports.y.value == 10


def test_interface_array_2d_index(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module top ports(
        x: interface async [2][3]Foo.In,
        y: interface async [3][2]Foo.Out,
    ) {
        comb {
            for (i in 0..2) {
                for (j in 0..3) {
                    y[j][i].value = x[i][j].value;
                }
            }
        }
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(2):
        for j in range(3):
            inst.ports.x[i][j].value.value = 3 * i + j
    inst.step(1)
    for i in range(2):
        for j in range(3):
            assert inst.ports.y[j][i].value.value == 3 * i + j


def test_interface_array_instance_for_loop(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module child ports(a: interface async Foo.In, b: interface async Foo.Out) {
        comb { b.value = a.value; }
    }
    module top ports(
        async {
            x: interface [4]Foo.In,
            y: interface [4]Foo.Out,
        }
    ) {
        for (i in 0..4) {
            instance child ports(a=x[i], b=y[3 - i]);
        }
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(4):
        inst.ports.x[i].value.value = i + 5
    inst.step(1)
    for i in range(4):
        assert inst.ports.y[3 - i].value.value == i + 5


def test_interface_array_instance_whole(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module child ports(a: interface async [2][2]Foo.In, b: interface async [2]Foo.Out) {
        comb {
            for (i in 0..2) {
                b[i].value = a[i][1].value;
            }
        }
    }
    module top ports(
        async {
            x: interface [3][2][2]Foo.In,
            y: interface [2]Foo.Out,
        }
    ) {
        instance child ports(a=x[1], b=y);
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(3):
        for j in range(2):
            for k in range(2):
                inst.ports.x[i][j][k].value.value = i + 2 * j + k
    inst.step(1)
    for j in range(2):
        assert inst.ports.y[j].value.value == 1 + 2 * j + 1


def test_interface_array_instance_shape_mismatch():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module child ports(a: interface async [2]Foo.In) {}
    module top ports(x: interface async [3]Foo.In) {
        instance child ports(a=x);
    }
    """
    with diag_error("interface array shape mismatch"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_instance_empty_interface_mismatch():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    interface Bar { value: uint(4), view In { value: in } }
    module child ports(a: interface async [0]Foo.In) {}
    module top ports(x: interface async [0]Bar.In) {
        instance child ports(a=x);
    }
    """
    with diag_error("interface mismatch"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_instance_single_vs_array():
    src_single_to_array = """
    interface Foo { value: uint(4), view In { value: in } }
    module child ports(a: interface async [1]Foo.In) {}
    module top ports(x: interface async Foo.In) {
        instance child ports(a=x);
    }
    """
    with diag_error("interface array shape mismatch"):
        _ = compile_custom(src_single_to_array).resolve_module("top.top")

    src_array_to_single = """
    interface Foo { value: uint(4), view In { value: in } }
    module child ports(a: interface async Foo.In) {}
    module top ports(x: interface async [1]Foo.In) {
        instance child ports(a=x);
    }
    """
    with diag_error("interface array shape mismatch"):
        _ = compile_custom(src_array_to_single).resolve_module("top.top")


def test_interface_array_instance_direction_mismatch():
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module child ports(a: interface async [2]Foo.In) {}
    module top ports(x: interface async [2]Foo.Out) {
        instance child ports(a=x);
    }
    """
    with diag_error("direction mismatch for interface port `value`"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_instance_domain_mismatch():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module child ports(clk: in clock, a: interface sync(clk) [2]Foo.In) {}
    module top ports(clk: in clock, x: interface async [2]Foo.In) {
        instance child ports(clk, a=x);
    }
    """
    with diag_error("invalid domain crossing: async to sync"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_generic_length(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module child(N: uint) ports(a: interface async [N]Foo.In, b: interface async [N]Foo.Out) {
        comb {
            for (i in 0..N) { b[i].value = a[i].value; }
        }
    }
    module top ports(x: interface async [3]Foo.In, y: interface async [3]Foo.Out) {
        instance child(3) ports(a=x, b=y);
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(3):
        inst.ports.x[i].value.value = i + 7
    inst.step(1)
    for i in range(3):
        assert inst.ports.y[i].value.value == i + 7


def test_interface_array_empty_slice():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module child ports(a: interface async [0]Foo.In) {}
    module top ports(x: interface async [4]Foo.In) {
        instance child ports(a=x[2..2]);
        instance child ports(a=x[4..]);
    }
    """
    _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_index_out_of_bounds():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(x: interface async [4]Foo.In, y: out async uint(4)) {
        comb { y = x[4].value; }
    }
    """
    with diag_error("array index out of bounds"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_index_hardware():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(async { x: interface [4]Foo.In, i: in uint(0..4), y: out uint(4) }) {
        comb { y = x[i].value; }
    }
    """
    with diag_error("feature not yet implemented: indexing interface arrays with a hardware value"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_slice_hardware():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(async { x: interface [4]Foo.In, i: in uint(0..2), y: out uint(4) }) {
        comb { y = x[i+..2][0].value; }
    }
    """
    with diag_error("feature not yet implemented: slicing interface arrays with a hardware start"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_as_value():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(x: interface async [4]Foo.In, y: out async uint(4)) {
        comb { y = x.value; }
    }
    """
    with diag_error("cannot access signal of interface array"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_array_slice(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module child ports(a: interface async [2]Foo.In, b: interface async [2]Foo.Out) {
        comb {
            for (i in 0..2) { b[i].value = a[i].value; }
        }
    }
    module top ports(
        async {
            x: interface [6]Foo.In,
            y: interface [6]Foo.Out,
        }
    ) {
        instance child ports(a=x[1..3], b=y[0..2]);
        instance child ports(a=x[3+..2], b=y[2..][0..2]);
        comb {
            y[4].value = x[4..][1].value;
            y[5].value = x[0..6][0..1][0].value;
        }
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(6):
        inst.ports.x[i].value.value = i + 1
    inst.step(1)
    expected = [2, 3, 4, 5, 6, 1]
    for i in range(6):
        assert inst.ports.y[i].value.value == expected[i]


def test_interface_array_slice_out_of_bounds():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(x: interface async [4]Foo.In, y: out async uint(4)) {
        comb { y = x[2..5][0].value; }
    }
    """
    with diag_error("array slice end out of bounds"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_wire_array(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module pass ports(a: interface async Foo.In, b: interface async Foo.Out) {
        comb { b.value = a.value; }
    }
    module pass_array ports(a: interface async [2][2]Foo.In, b: interface async [2][2]Foo.Out) {
        comb {
            for (i in 0..2) {
                for (j in 0..2) { b[i][j].value = a[i][j].value; }
            }
        }
    }
    module top ports(
        async {
            x: interface [2][2]Foo.In,
            y: interface [2][2]Foo.Out,
        }
    ) {
        wire v: interface [2][2]Foo;
        wire w: async interface [2][2]Foo;
        instance pass_array ports(a=x, b=v);
        for (i in 0..2) {
            for (j in 0..2) {
                instance pass ports(a=v[i][j], b=w[1 - i][j]);
            }
        }
        instance pass_array ports(a=w, b=y);
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(2):
        for j in range(2):
            inst.ports.x[i][j].value.value = 2 * i + j + 1
    inst.step(1)
    for i in range(2):
        for j in range(2):
            assert inst.ports.y[1 - i][j].value.value == 2 * i + j + 1


def test_interface_wire_array_domain_per_element():
    src_ok = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module source ports(clk: in clock, a: interface sync(clk) Foo.Out) {
        comb { a.value = 0; }
    }
    module top ports(clk_a: in clock, clk_b: in clock) {
        wire w: interface [2]Foo;
        instance source ports(clk=clk_a, a=w[0]);
        instance source ports(clk=clk_b, a=w[1]);
    }
    """
    _ = compile_custom(src_ok).resolve_module("top.top")

    src_err = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module source ports(clk: in clock, a: interface sync(clk) Foo.Out) {
        comb { a.value = 0; }
    }
    module sink ports(clk: in clock, a: interface sync(clk) Foo.In) {}
    module top ports(clk_a: in clock, clk_b: in clock) {
        wire w: interface [2]Foo;
        instance source ports(clk=clk_a, a=w[0]);
        instance sink ports(clk=clk_b, a=w[0]);
    }
    """
    with diag_error("invalid domain crossing: different clock"):
        _ = compile_custom(src_err).resolve_module("top.top")


def test_interface_wire_array_whole_shape_mismatch():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module child ports(a: interface async [2]Foo.In) {}
    module top ports() {
        wire w: interface [3]Foo;
        instance child ports(a=w);
    }
    """
    with diag_error("interface array shape mismatch"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_wire_not_interface():
    src = """
    module top ports() {
        wire w: interface bool;
    }
    """
    with diag_error("expected interface or interface array"):
        _ = compile_custom(src).resolve_module("top.top")


def test_interface_typeof():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(x: interface async [2]Foo.In) {
        type t = type(x);
    }
    """
    with diag_error("interface instances do not have a type"):
        _ = compile_custom(src).resolve_module("top.top")
