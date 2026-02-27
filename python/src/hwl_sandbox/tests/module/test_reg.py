from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_reg_simple():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        wire w: uint(8);
        clocked(clk, async rst) {
            reg port y = false;
            reg wire w = 5;
            reg r = true;
            
            y = !y;
            w = 0;
            r = r ^ y;
        }
    }
    """
    _ = compile_custom(src).resolve("top.top")


def test_reg_port_simple():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        clocked(clk, async rst) {
            reg port y = false;
            y = !y;
        }
    }
    """
    _ = compile_custom(src).resolve("top.top")


def test_non_reg_clocked():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        clocked(clk, async rst) {
            y = !y;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="clocked process cannot drive port"):
        _ = compile_custom(src).resolve("top.top")


def test_reg_twice():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        clocked(clk, async rst) {
            reg port y = false;
            reg port y = false;
            y = !y;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="port already marked as a register in this process"):
        _ = compile_custom(src).resolve("top.top")


def test_reg_domain_different():
    src = """
    // different domain, but valid: output domain is less strict
    module top_valid ports(clk: in clock, rst: in async bool, y: out async bool) {
        clocked(clk, async rst) {
            reg port y = false;
        }
    }
    
    // different domain, invalid: output domain is more strict
    module top_invalid ports(clk: in clock, rst: in async bool, y: out sync(clk) bool) {
        clocked(clk, async rst) {
            reg port y = false;
        }
    }
    """
    _ = compile_custom(src).resolve("top.top_valid")
    with pytest.raises(hwl.DiagnosticException, match="invalid domain crossing"):
        _ = compile_custom(src).resolve("top.top_invalid")


def test_reg_decl_in_function(tmp_dir: Path):
    src = """
    module top ports(
        clk: in clock,
        rst: in async bool,
        sync(clk, async rst) {
            x: in int(8),
            y1: out int(8),
            y2: out int(8),
            y3: out int(8),
        }
    ) {
        clocked(clk, async rst) {
            reg port y1 = -1;
            reg port y2 = -1;
            reg port y3 = -1;
        
            y1 = x;
            y2 = delay(int(8), x, -1);
            y3 = delay(int(8), delay(int(8), x, -1), -1);
        }
    }
    
    fn delay(T: type, next: T, reset: T) -> T {
        reg d: T = reset;
        val v = d;
        d = next;
        return v;
    }
    """
    top = compile_custom(src).resolve("top.top")

    print(top.as_verilog().source)
    inst = top.as_verilated(tmp_dir).instance()

    ports = inst.ports

    ports.rst.value = False
    inst.step(1)
    ports.rst.value = True
    inst.step(1)
    ports.rst.value = False
    inst.step(1)

    values = range(4)
    for i in values:
        ports.x.value = i

        ports.clk.value = True
        inst.step(1)
        ports.clk.value = False
        inst.step(1)

        expected = ([-1] * 2 + list(values))[i:i + 3]
        actual = [ports.y3.value, ports.y2.value, ports.y1.value]
        assert expected == actual
