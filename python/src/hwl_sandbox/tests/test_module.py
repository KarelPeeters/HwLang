from pathlib import Path

from hwl import hwl

from hwl_sandbox.common.util import compile_custom


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
