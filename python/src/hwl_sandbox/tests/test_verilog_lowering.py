"""Tests for Verilog code generation / lowering."""

import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_bool_literal_lowering():
    """Test that boolean literals are correctly lowered to Verilog."""
    src = """
    import std.types.bool;

    pub module test_bool_literals ports(
        x_true: out async bool,
        x_false: out async bool
    ) {
        comb {
            x_true = true;
            x_false = false;
        }
    }
    """
    compile = compile_custom(src)
    module = compile.resolve("top.test_bool_literals")
    verilog = module.as_verilog()
    
    # Check that true is lowered to 1'b1
    assert "x_true = 1'b1" in verilog.source, \
        f"Expected 'x_true = 1'b1' in Verilog output but got:\n{verilog.source}"
    
    # Check that false is lowered to 1'b0
    assert "x_false = 1'b0" in verilog.source, \
        f"Expected 'x_false = 1'b0' in Verilog output but got:\n{verilog.source}"
