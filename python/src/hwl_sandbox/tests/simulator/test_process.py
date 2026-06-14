from hwl_sandbox.common.util import BuildSim, compile_custom


def test_process_comb_basic(build_sim: BuildSim):
    src = """
    type T = uint(8);
    module top ports(x: in async T, y: out async T) {
        comb { y = x; }
    }
    """
    top = compile_custom(src).resolve_module("top.top")
    inst = build_sim(top).instance()

    for i in range(8):
        inst.ports.x.value = i
        inst.step(1)
        assert inst.ports.y.value == i


def test_process_clocked_basic(build_sim: BuildSim):
    src = """
    type T = uint(8);
    module top ports(clk: in clock, rst: in async bool, sync(clk, async rst) { x: in T, y: out T }) {
        clocked(clk, async rst) {
            reg wire y = 255;
            y = x;
        }
    }
    """

    top = compile_custom(src).resolve_module("top.top")
    inst = build_sim(top).instance()

    # reset
    inst.step(1)
    inst.ports.rst.value = True
    inst.step(1)
    inst.ports.rst.value = False
    inst.step(1)

    # check that reset worked
    y_prev = 255
    assert inst.ports.y.value == y_prev

    # clock cycles
    for i in range(8):
        inst.ports.x.value = i

        # check that just stepping without a clock edge does not propgate yet
        inst.step(1)
        assert inst.ports.y.value == y_prev

        # clock edge
        inst.ports.clk.value = True
        inst.step(1)
        inst.ports.clk.value = False
        inst.step(1)

        # now the value should have been updated
        assert inst.ports.y.value == i
        y_prev = i
