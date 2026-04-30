from pathlib import Path

from hwl_sandbox.common.util import compile_custom


def test_cpp_backend_clocked_process(tmp_dir: Path):
    src = """
    module top ports(clk: in clock, rst: in async bool, x: in sync(clk, async rst) uint(8), y: out sync(clk, async rst) uint(8)) {
        clocked(clk, async rst) {
            reg wire y = 0;
            y = x;
        }
    }
    """
    top = compile_custom(src).resolve("top.top")
    build_dir = tmp_dir / "cpp"
    build_dir.mkdir()
    inst = top.as_cpp(build_dir).instance()

    ports = inst.ports
    ports.rst.value = False
    ports.clk.value = False
    ports.x.value = 0
    inst.step(1)

    ports.rst.value = True
    inst.step(1)
    assert ports.y.value == 0

    ports.rst.value = False
    ports.x.value = 7
    inst.step(1)
    ports.clk.value = True
    inst.step(1)
    assert ports.y.value == 7

    ports.clk.value = False
    inst.step(1)
    ports.x.value = 11
    inst.step(1)
    ports.clk.value = True
    inst.step(1)
    assert ports.y.value == 11


def test_cpp_backend_reset_wins_over_clock_edge(tmp_dir: Path):
    src = """
    module top ports(clk: in clock, rst: in async bool, x: in sync(clk, async rst) uint(8), y: out sync(clk, async rst) uint(8)) {
        clocked(clk, async rst) {
            reg wire y = 0;
            y = x;
        }
    }
    """
    top = compile_custom(src).resolve("top.top")
    build_dir = tmp_dir / "cpp"
    build_dir.mkdir()
    inst = top.as_cpp(build_dir).instance()

    ports = inst.ports
    ports.clk.value = False
    ports.rst.value = False
    ports.x.value = 7
    inst.step(1)
    ports.clk.value = True
    inst.step(1)
    assert ports.y.value == 7

    ports.clk.value = False
    ports.rst.value = True
    ports.x.value = 13
    inst.step(1)
    ports.clk.value = True
    inst.step(1)
    assert ports.y.value == 0


def test_cpp_backend_clocked_assignments_are_immediately_visible(tmp_dir: Path):
    src = """
    module top ports(clk: in clock, rst: in async bool, x: in sync(clk, async rst) uint(8), y: out sync(clk, async rst) uint(8), z: out sync(clk, async rst) uint(8)) {
        clocked(clk, async rst) {
            reg wire y = 0;
            reg wire z = 0;
            y = x;
            z = y;
        }
    }
    """
    top = compile_custom(src).resolve("top.top")
    build_dir = tmp_dir / "cpp"
    build_dir.mkdir()
    inst = top.as_cpp(build_dir).instance()

    ports = inst.ports
    ports.clk.value = False
    ports.rst.value = False
    ports.x.value = 7
    inst.step(1)
    ports.clk.value = True
    inst.step(1)

    assert ports.y.value == 7
    assert ports.z.value == 7
