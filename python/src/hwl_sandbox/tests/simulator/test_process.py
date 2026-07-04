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
        assert inst.ports.x.value == i
        if i > 0:
            assert inst.ports.y.value == i - 1

        inst.step(1)
        assert inst.ports.y.value == i


def test_process_clocked_basic_simulator(build_sim: BuildSim):
    # test the very specific details of the simulator event system
    src = """
    type T = uint(8);
    module top ports(clk: in clock, rst: in async bool, sync(clk, async rst) { x: in T, y: out T }) {
        clocked(clk, async rst) {
            print("clock tick");
            reg wire y = 255;
            y = x;
        }
    }
    """
    y_reset = 255

    top = compile_custom(src).resolve_module("top.top")
    inst = build_sim(top).instance()

    # reset
    inst.step(1)
    inst.ports.rst.value = True
    inst.step(1)
    assert inst.ports.y.value == y_reset
    inst.ports.rst.value = False
    inst.step(1)
    assert inst.ports.y.value == y_reset

    # clock cycles
    y_prev = y_reset
    for i in range(8):
        print(f"{i=}")
        inst.ports.x.value = i

        # check that just stepping without a clock edge does not propagate yet
        # inst.step(1)
        # assert inst.ports.y.value == y_prev

        # clock edge
        inst.ports.clk.value = True
        inst.step(1)
        inst.ports.clk.value = False
        inst.step(1)

        # now the value should have been updated
        # TODO move this right after the posedge
        assert inst.ports.y.value == i
        y_prev = i


# TODO add asserts that check that values are undef initially
# TODO test that there is a reasonable sequence of calls that works on both simulators
# TODO test that multiple clocked processes correctly see the previous values of each other
# TODO test clocked/comb process interaction, in both directions
# TODO check simultaneous going out of reset and clock edge
# TODO check simultaneous changing of multiple clocks
# TODO check non-simul changing of multiple clocks
# TODO test that changing data input signals by themselves does nothing


if __name__ == '__main__':
    # print("derp")
    test_process_clocked_basic_simulator(lambda m: m.as_simulator())
