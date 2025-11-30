from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


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


def test_simple_module_instance(tmp_dir: Path):
    src = """
    import std.types.bool;
    module parent ports(x: in async bool, y: out async bool) { instance child ports(x, y); }
    module child ports(x: in async bool, y: out async bool) { comb { y = x; } }
    """

    c = compile_custom(src)
    parent: hwl.Module = c.resolve("top.parent")
    print(parent.as_verilog().source)
    parent_verilated: hwl.ModuleVerilated = parent.as_verilated(tmp_dir)
    parent_inst: hwl.VerilatedInstance = parent_verilated.instance()

    for v in [False, True, False, True]:
        parent_inst.ports.x.value = v
        parent_inst.step(1)
        assert parent_inst.ports.x.value == v
