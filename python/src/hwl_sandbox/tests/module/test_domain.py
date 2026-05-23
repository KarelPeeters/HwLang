from hwl_sandbox.common.util import compile_custom, diag_error


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

# TODO test some valid/invalid domain crossings,
#    first in assignments in comb/clocked blocks,
#    but also in child module ports and wire init values
