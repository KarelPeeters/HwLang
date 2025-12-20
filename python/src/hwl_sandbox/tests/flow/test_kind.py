import pytest

from hwl_sandbox.common.util import compile_custom


def test_flow_read_hw_in_hw():
    src = """import std.types.bool; module foo ports(p: in async bool) { comb { val _ = p; } }"""
    compile_custom(src).resolve("top.foo")


def test_flow_read_hw_in_const():
    src = """import std.types.bool; module foo ports(p: in async bool) { const { val _ = p + 2; } }"""
    with pytest.raises(match="signal evaluation is only allowed in a hardware context"):
        compile_custom(src).resolve("top.foo")
