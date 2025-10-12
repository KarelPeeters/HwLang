from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


def test_port_reg_name(tmpdir: Path):
    src = """
    import std.types.bool;
    module foo ports(clk: in clock, a: out sync(clk) bool) {
        reg out a = undef;
        clocked(clk) {
            a = !a;
        }
    }
    """

    c = compile_custom(src)
    m: hwl.Module = c.resolve("top.foo")
    m.as_verilated(tmpdir)
