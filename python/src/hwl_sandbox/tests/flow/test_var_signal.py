from pathlib import Path

from hwl_sandbox.common.util import compile_custom


def test_read_const_from_signal(tmp_dir: Path):
    src = """
    import std.types.uint;
    module foo ports(clk: in clock, p: out async uint(8)) {
        wire w: uint(8);
        reg r: uint(8) = undef;
        comb {
            w = 3;
            p = 4;
            const _ = p;
            const _ = w;
        }
        clocked(clk) {
            r = 5;
            const _ = r;
        }
    }
    """
    c = compile_custom(src)
    _ = c.resolve("top.foo")


def test_read_range_from_signal(tmp_dir: Path):
    src = """
    import std.types.uint;
    module foo ports(p: out async uint(8), q: in async uint(4)) {
        comb {
            p = q;
            val _: uint(4) = p;
        }
    }
    """
    c = compile_custom(src)
    _ = c.resolve("top.foo")
