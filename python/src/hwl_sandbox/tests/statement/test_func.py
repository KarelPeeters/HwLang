import hwl

from hwl_sandbox.common.util import compile_custom


def test_func_capture_const():
    src = """
    const c = 5;
    fn f() -> any {
        fn g() -> int { 
            return c;
        }
        return g;
    }
    """
    f: hwl.Function = compile_custom(src).resolve("top.f")
    assert f()() == 5


def test_func_capture_param():
    src = """
    fn f(x: int) -> any {
        fn g() -> int {
            return x;
        }
        return g;
    }
    """
    f: hwl.Function = compile_custom(src).resolve("top.f")
    for x in [4, 8]:
        assert f(x)() == x


def test_func_capture_iter_var():
    src = """
    fn f(n: uint) -> any {
        var r = [];
        for (i in 0..n) {
            fn g() -> int {
                return i;
            }
            r = [*r, g];
        }
        return r;
    }
    """
    f: hwl.Function = compile_custom(src).resolve("top.f")

    n = 4
    r = f(n)
    for i in range(n):
        assert r[i]() == i
