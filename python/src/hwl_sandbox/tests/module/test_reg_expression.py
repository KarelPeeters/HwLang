from pathlib import Path

from hwl_sandbox.common.util import compile_custom


def test_reg_expression(tmp_dir: Path):
    src = """
    module top ports(
        clk: in clock,
        rst: in async bool,
        sync(clk, rst) {
            x: in int(8),
            y1: out int(8),
            y2: out int(8),
            y3: out int(8),
        }
    ) {
        reg out y1 = -1;
        reg out y2 = -1;
        reg out y3 = -1;
        clocked(clk, async rst) {
            y1 = x;
            y2 = reg(x, -1);
            y3 = reg(reg(x, -1), -1);
        }
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
