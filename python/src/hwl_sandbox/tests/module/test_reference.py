from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_ref_port_read(tmp_dir: Path):
    src = """
    module top ports(x: in async uint(8), y: out async uint(8)) {
        comb {
            val r = ref x;
            y = deref r;            
        }
    }
    """
    top: hwl.Module = compile_custom(src).resolve("top.top")
    inst = top.as_verilated(tmp_dir).instance()
    for v in [0, 4]:
        inst.ports.x.value = v
        inst.step(1)
        assert inst.ports.y.value == v


def test_ref_port_write(tmp_dir: Path):
    src = """
    module top ports(x: in async uint(8), y: out async uint(8)) {
        comb {
            val r = ref y;
            deref r = x;            
        }
    }
    """
    top: hwl.Module = compile_custom(src).resolve("top.top")
    inst = top.as_verilated(tmp_dir).instance()
    for v in [0, 4]:
        inst.ports.x.value = v
        inst.step(1)
        assert inst.ports.y.value == v


def test_ref_port_write_input_error():
    src = """
    module top ports(x: in async uint(8), y: in async uint(8)) {
        comb {
            val r = ref y;
            deref r = x;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="cannot assign to input port"):
        compile_custom(src).resolve("top.top")


def test_ref_interface():
    src = """
    interface Pair {
        x: uint(8), y: uint(8),
        interface Mixed { x: in, y: out }
    }
    module top_basic ports(p: interface async Pair.Mixed) {
        comb {
            val _ = p.x;
            p.y = 0;
        }
    }
    module top_ref ports(p: interface async Pair.Mixed) {
        comb {
            val v = ref p;
            val _ = (deref v).x;
            (deref v).y = 0;
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
    with pytest.raises(hwl.DiagnosticException, match="cannot evaluate interface as value"):
        _ = c.resolve("top.top_wrong")


def test_ref_ty():
    src = """
    interface Data {
        x: uint(8),
        interface View { x: in }
    }
    module top_correct ports(p_single: in async uint(8), p_intf: interface async Data.View) {
        comb {
            val r_single: Ref(uint(8)) = ref p_single;
            val r_intf: RefInterface(Data) = ref p_intf;
        }
    }
    module top_wrong_single ports(p_single: in async uint(8), p_intf: interface async Data.View) {
        comb {
            val r_single: uint(8) = ref p_single;
        }
    }
    module top_wrong_intf ports(p_single: in async uint(8), p_intf: interface async Data.View) {
        comb {
            val r_single: uint(8) = ref p_intf;
        }
    }
    """
    compile_custom(src).resolve("top.top_correct")
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        compile_custom(src).resolve("top.top_wrong_single")
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        compile_custom(src).resolve("top.top_wrong_intf")


def test_ref_var():
    src = """
    fn f(x: uint) -> uint {
        var v = 0;
        val w = ref v;
        deref w = x;
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
            w = ref v;
        }
        return deref w;
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="cannot access variable after its scope has ended"):
        f(4)


def test_ref_var_write_dropped():
    src = """
    fn f(x: uint) -> uint {
        val w;
        if (true) {
            var v = x;
            w = ref v;
        }
        deref w = 4;
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="cannot access variable after its scope has ended"):
        f(4)
