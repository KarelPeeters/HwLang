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
        interface mixed { x: in, y: out }
    }
    module top_basic ports(p: interface async Pair.mixed) {
        comb {
            val _ = p.x;
            p.y = 0;
        }
    }
    module top_ref ports(p: interface async Pair.mixed) {
        comb {
            val v = ref p;
            val _ = (deref v).x;
            (deref v).y = 0;
        }
    }
    module top_wrong ports(p: interface async Pair.mixed) {
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
