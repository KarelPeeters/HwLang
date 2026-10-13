from khdl_sandbox.common.util import compile_custom, diag_error


# TODO test some valid/invalid domain crossings,
#    first in assignments in comb/clocked blocks,
#    but also in child module ports and wire init values

def test_domain_unsafe():
    src = """
    module top(c: bool) ports(clk: in clock, x: in async bool, y: out sync(clk) bool) {
        comb {
            if (c) {
                y = unsafe_value_with_domain(x, sync(clk));
            } else {
                y = x;
            }
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    with diag_error("invalid domain crossing: async to sync"):
        _ = top(c=False)
    _ = top(c=True)


TRANSITIONS = [
    ("sync(clk)", "sync(clk)", True),
    ("sync(clk)", "async", True),
    ("async", "sync(clk)", False),
    ("async", "async", True),
]


def test_domain_instance_simple_input():
    for dp, dc, valid in TRANSITIONS:
        src = f"""
        module parent ports(clk: in clock, x: in {dp} bool) {{ instance child ports(clk, x); }}
        module child ports(clk: in clock, x: in {dc} bool) {{}}
        """
        c = compile_custom(src)
        if valid:
            _ = c.resolve_module("top.parent")
        else:
            with diag_error(f"invalid domain crossing: async to sync"):
                _ = c.resolve_module("top.parent")


def test_domain_instance_simple_output():
    for dp, dc, valid in TRANSITIONS:
        src = f"""
        module parent ports(clk: in clock, y: out {dc} bool) {{ instance child ports(clk, y); }}
        module child ports(clk: in clock, y: out {dp} bool) {{ comb {{ y = false; }} }}
        """
        c = compile_custom(src)
        if valid:
            _ = c.resolve_module("top.parent")
        else:
            with diag_error(f"invalid domain crossing: async to sync"):
                _ = c.resolve_module("top.parent")


def test_domain_instance_interface_input():
    for dp, dc, valid in TRANSITIONS:
        src = f"""
        interface I {{ d: bool, view In {{ d: in }} }}
        module parent ports(clk: in clock, x: interface {dp} I.In) {{ instance child ports(clk, x); }}
        module child ports(clk: in clock, x: interface {dc} I.In) {{}}
        """
        c = compile_custom(src)
        if valid:
            _ = c.resolve_module("top.parent")
        else:
            with diag_error(f"invalid domain crossing: async to sync"):
                _ = c.resolve_module("top.parent")


def test_domain_instance_interface_output():
    for dp, dc, valid in TRANSITIONS:
        src = f"""
        interface I {{ d: bool, view Out {{ d: out }} }}
        module parent ports(clk: in clock, x: interface {dc} I.Out) {{ instance child ports(clk, x); }}
        module child ports(clk: in clock, x: interface {dp} I.Out) {{ comb {{ x.d = false; }} }}
        """
        c = compile_custom(src)
        if valid:
            _ = c.resolve_module("top.parent")
        else:
            with diag_error(f"invalid domain crossing: async to sync"):
                _ = c.resolve_module("top.parent")
