from hwl_sandbox.common.util import compile_custom, diag_error


def test_assign_left():
    src = "fn f() { var v = 0; v = 8; }"
    f = compile_custom(src).resolve("top.f")
    f()


def test_assign_right():
    src = "fn f() { 4 = 8; }"
    f = compile_custom(src).resolve("top.f")
    with diag_error("cannot use value as target"):
        f()


def test_assign_interface():
    src = """
    interface Foo {
        interface View {}
    }
    module top ports(x: interface async Foo.View) {
        comb {
            x = 8;
        }
    }
    """
    with diag_error("cannot use interface as target"):
        _ = compile_custom(src).resolve_module("top.top")
