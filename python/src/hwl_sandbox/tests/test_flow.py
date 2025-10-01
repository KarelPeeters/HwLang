import pytest

from hwl_sandbox.common.util import compile_custom


def test_flow_read_hw_in_hw():
    src = """import std.types.bool; module foo ports(p: in async bool) { comb { val _ = p; } }"""
    compile_custom(src).resolve("top.foo")


def test_flow_read_hw_in_const():
    src = """import std.types.bool; module foo ports(p: in async bool) { const { val _ = p + 2; } }"""
    with pytest.raises(match="port access is only allowed in a hardware context"):
        compile_custom(src).resolve("top.foo")


def test_conditional_write():
    src = """
    import std.types.[bool, int];
    module foo ports(x: in async int(0..8)) {
        comb {
            var a = false;
            if (x < 4) {
                a = true;
            }
        }
    }
    """
    compile_custom(src).resolve("top.foo")


def test_inline_register():
    # TODO simulate and check that this results in two cycles of delay
    src = """
    import std.types.[bool, int];
    module foo ports(clk: in clock, x: in sync(clk) bool, y: out sync(clk) bool) {
        reg out y = undef;
        clocked(clk) {
            y = reg(x, undef);
        }
    }
    """
    compile_custom(src).resolve("top.foo")
