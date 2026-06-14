from hwl_sandbox.common.util import BuildSim, compile_custom


def test_schedule_shuffled_comb(build_sim: BuildSim):
    # test scheduling through a bunch of combinatorial processes,
    #   intentionally re-ordered so simple ordered scheduling would not be correct
    src = """
    type T = uint(8);
    module top ports(x: in async T, y: out async T) {
        wire w0: async T;
        wire w1: async T;
        wire w2: async T;
        wire w3: async T;
        
        comb { w0 = x; }
        comb { y = w3; }
        comb { w2 = w1; }
        comb { w1 = w0; }
        comb { w3 = w2; }
    }
    """
    top = compile_custom(src).resolve_module("top.top")
    inst = build_sim(top).instance()

    for i in range(8):
        inst.ports.x.value = i
        inst.step(1)
        assert inst.ports.y.value == i


def test_schedule_different_children(build_sim: BuildSim):
    # test scheduling through child modules,
    #   intentionally re-used in multiple ways so per-instance scheduling is necessary
    src = """
    type T = uint(8);
    module top ports(x: in async T, y: out async T) {
        wire w0: async T;
        wire w1: async T;
        wire w2: async T;
        
        instance child_pair ports(x0=x, y0=w0, x1=w2, y1=y);
        instance child_pair ports(x0=w1, y0=w2, x1=w0, y1=w1);
    }
    module child_pair ports(x0: in async T, y0: out async T, x1: in async T, y1: out async T) {
        instance child_single ports(x=x0, y=y0);
        instance child_single ports(x=x1, y=y1);
    }
    module child_single ports(x: in async T, y: out async T) {
        comb { y = x; }
    }
    """
    top = compile_custom(src).resolve_module("top.top")
    inst = build_sim(top).instance()

    for i in range(8):
        inst.ports.x.value = i
        inst.step(1)
        assert inst.ports.y.value == i


def test_schedule_through_dummy_output_port(build_sim: BuildSim):
    src = """
    type T = uint(8);
    module top ports(x0: in async T, x1: in async T, y0: out async T, y1: out async T) {
        instance child ports(x=x0, m=_, y=y0);
        instance child ports(x=x1, m=_, y=y1);
    }
    module child ports(x: in async T, m: out async T, y: out async T) {
        comb { y = m; }
        comb { m = x; }
    }
    """
    top = compile_custom(src).resolve_module("top.top")
    inst = build_sim(top).instance()

    for i in range(8):
        inst.ports.x0.value = i
        inst.ports.x1.value = 8 - i
        inst.step(1)
        assert inst.ports.y0.value == i
        assert inst.ports.y1.value == 8 - i
