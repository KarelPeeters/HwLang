from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


def test_read_const_from_signal(tmp_dir: Path):
    src = """
    import std.types.uint;
    module top ports(clk: in clock, p: out async uint(8)) {
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
    _ = c.resolve("top.top")


def test_read_range_from_signal(tmp_dir: Path):
    src = """
    import std.types.uint;
    module top ports(p: out async uint(8), q: in async uint(4)) {
        comb {
            p = q;
            val _: uint(4) = p;
        }
    }
    """
    c = compile_custom(src)
    _ = c.resolve("top.top")


# TODO test that signals are immediately read back instead of only the next cycle
# def test_signal_readback(tmp_dir: Path):
#     src = """
#     import std.types.[bool, int, uint];
#     module top ports(clk: in clock, x: in sync(clk) uint(8), y: out sync(clk) uint(8)) {
#         reg out y = undef;
#         clocked(clk) {
#             y = x;
#             y = 255 - y;
#         }
#     }
#     """
#     c = compile_custom(src)
#     foo: hwl.Module = c.resolve("top.top")
#     print(foo.as_verilog())
#     foo_inst = foo.as_verilated(tmp_dir).instance()
#
#     ports = foo_inst.ports
#
#     def cycle():
#         foo_inst.step(1)
#         ports.clk.value = 1
#         foo_inst.step(1)
#         ports.clk.value = 0
#
#     ports.clk.value = 0
#     ports.x.value = 0
#     cycle()
#     assert ports.y.value == 255


def test_write_after_read_var(tmp_dir: Path):
    src = """
    import std.types.uint;
    module top ports(x: in async uint(8), y: out async uint(16)) {
        comb {
            var v = x;
            val w = v + {
                v = 0;
                1
            };
            y = w;
        }
    }
    """
    check_write_after_read(tmp_dir, src)


def test_write_after_read_var_array(tmp_dir: Path):
    src = """
    import std.types.uint;
    module top ports(x: in async uint(8), y: out async uint(16)) {
        comb {
            var v: [1]uint(8) = [x];
            val w = v[0] + {
                v[0] = 0;
                1
            };
            y = w;
        }
    }
    """
    check_write_after_read(tmp_dir, src)


def test_write_after_read_wire(tmp_dir: Path):
    src = """
    import std.types.uint;
    module top ports(x: in async uint(8), y: out async uint(16)) {
        wire w: uint(8);
        comb {
            w = x;
            val v = w + {
                w = 0;
                1
            };
            y = v;
        }
    }
    """
    check_write_after_read(tmp_dir, src)


def test_write_after_read_wire_array(tmp_dir: Path):
    src = """
    import std.types.uint;
    module top ports(x: in async uint(8), y: out async uint(16)) {
        wire w: [1]uint(8);
        comb {
            w[0] = x;
            val v = w[0] + {
                w[0] = 0;
                1
            };
            y = v;
        }
    }
    """
    check_write_after_read(tmp_dir, src)


def check_write_after_read(tmp_dir: Path, src: str):
    c = compile_custom(src)
    foo: hwl.Module = c.resolve("top.top")
    print(foo.as_verilog().source)
    foo_inst = foo.as_verilated(tmp_dir).instance()

    foo_inst.ports.x.value = 4
    foo_inst.step(1)
    assert foo_inst.ports.y.value == 5


def test_const_assign_outside_var():
    src = """
    import std.types.bool;
    import std.util.assert;
    module top ports(c: in async bool) {
        comb {
            var v = c;
            const {
                v = true;
            }
            const {
                assert(v);
            }
        }
    }
    """
    c = compile_custom(src)
    _ = c.resolve("top.top")
