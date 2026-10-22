from pathlib import Path

import khdl

from khdl_sandbox.common.util import compile_custom, diag_error


def test_ref_port_read(tmp_dir: Path):
    src = """
    module top ports(x: in async uint(8), y: out async uint(8)) {
        comb {
            val r = ref(x);
            y = deref(r);            
        }
    }
    """
    top: khdl.Module = compile_custom(src).resolve("top.top")
    inst = top.as_verilated(tmp_dir).instance()
    for v in [0, 4]:
        inst.ports.x.value = v
        inst.step(1)
        assert inst.ports.y.value == v


def test_ref_port_write(tmp_dir: Path):
    src = """
    module top ports(x: in async uint(8), y: out async uint(8)) {
        comb {
            val r = ref(y);
            deref(r) = x;            
        }
    }
    """
    top: khdl.Module = compile_custom(src).resolve("top.top")
    inst = top.as_verilated(tmp_dir).instance()
    for v in [0, 4]:
        inst.ports.x.value = v
        inst.step(1)
        assert inst.ports.y.value == v


def test_ref_port_write_input_error():
    src = """
    module top ports(x: in async uint(8), y: in async uint(8)) {
        comb {
            val r = ref(y);
            deref(r) = x;
        }
    }
    """
    with diag_error("cannot assign to input port"):
        compile_custom(src).resolve("top.top")


def test_ref_interface():
    src = """
    interface Pair {
        x: uint(8), y: uint(8),
        view Mixed { x: in, y: out }
    }
    module top_basic ports(p: interface async Pair.Mixed) {
        comb {
            val _ = p.x;
            p.y = 0;
        }
    }
    module top_ref ports(p: interface async Pair.Mixed) {
        comb {
            val v = ref(p);
            val _ = deref(v).x;
            deref(v).y = 0;
        }
    }
    module top_wrong ports(p: interface async Pair.Mixed) {
        comb {
            val v = p;
        }
    }
    """
    c = compile_custom(src)

    _ = c.resolve("top.top_basic")
    _ = c.resolve("top.top_ref")
    with diag_error("cannot evaluate interface as value"):
        _ = c.resolve("top.top_wrong")


def test_ref_ty():
    src = """
    interface Data {
        x: uint(8),
        view View { x: in }
    }
    module top_correct ports(p_single: in async uint(8), p_intf: interface async Data.View) {
        comb {
            val r_single: Ref(uint(8)) = ref(p_single);
            val r_intf: RefInterface(Data) = ref(p_intf);
        }
    }
    module top_wrong_single ports(p_single: in async uint(8), p_intf: interface async Data.View) {
        comb {
            val r_single: uint(8) = ref(p_single);
        }
    }
    module top_wrong_intf ports(p_single: in async uint(8), p_intf: interface async Data.View) {
        comb {
            val r_single: uint(8) = ref(p_intf);
        }
    }
    """
    compile_custom(src).resolve("top.top_correct")
    with diag_error("type mismatch"):
        compile_custom(src).resolve("top.top_wrong_single")
    with diag_error("type mismatch"):
        compile_custom(src).resolve("top.top_wrong_intf")


def test_ref_var():
    src = """
    fn f(x: uint) -> uint {
        var v = 0;
        val w = ref(v);
        deref(w) = x;
        return v;
    }
    """
    f = compile_custom(src).resolve("top.f")
    assert f(4) == 4


def test_ref_var_read_dropped():
    src = """
    fn f(x: uint) -> uint {
        val w;
        if (true) {
            var v = x;
            w = ref(v);
        }
        return deref(w);
    }
    """
    f = compile_custom(src).resolve("top.f")
    with diag_error("cannot access variable after its scope has ended"):
        f(4)


def test_ref_var_write_dropped():
    src = """
    fn f(x: uint) -> uint {
        val w;
        if (true) {
            var v = x;
            w = ref(v);
        }
        deref(w) = 4;
    }
    """
    f = compile_custom(src).resolve("top.f")
    with diag_error("cannot access variable after its scope has ended"):
        f(4)


def test_ref_interface_array(tmp_dir: Path):
    src = """
    interface Foo { value: uint(4), view In { value: in } view Out { value: out } }
    module child ports(a: interface async [2]Foo.In, b: interface async [2]Foo.Out) {
        comb {
            for (i in 0..2) { b[i].value = a[i].value; }
        }
    }
    module top ports(x: interface async [3][2]Foo.In, y: interface async [2]Foo.Out) {
        const rx: RefInterface([3][2]Foo) = ref(x);
        const ry: RefInterface([2]Foo) = ref(y);
        instance child ports(a=deref(rx)[2], b=deref(ry));
    }
    """
    m = compile_custom(src).resolve_module("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for i in range(3):
        for j in range(2):
            inst.ports.x[i][j].value.value = 2 * i + j
    inst.step(1)
    for j in range(2):
        assert inst.ports.y[j].value.value == 4 + j


def test_ref_interface_array_ty_mismatch():
    src = """
    interface Foo { value: uint(4), view In { value: in } }
    module top ports(x: interface async [3]Foo.In) {
        const r: RefInterface([2]Foo) = ref(x);
    }
    """
    with diag_error("type mismatch"):
        compile_custom(src).resolve("top.top")
