from hwl_sandbox.common.util import compile_custom, diag_error


def test_flow_read_hw_in_hw():
    src = """module foo ports(p: in async bool) { comb { val _ = p; } }"""
    compile_custom(src).resolve("top.foo")


def test_flow_read_hw_in_const():
    src = """module foo ports(p: in async bool) { const { val _ = p + 2; } }"""
    with diag_error("signal evaluation is only allowed in a hardware context"):
        compile_custom(src).resolve("top.foo")
