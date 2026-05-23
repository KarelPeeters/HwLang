from hwl_sandbox.common.util import compile_custom, diag_error


def test_dummy():
    src = "const c = _;"
    with diag_error("dummy expression not allowed in this context"):
        _ = compile_custom(src).resolve("top.c")
