import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_normal_struct():
    c = compile_custom("struct S { a: int, b: int }")
    _ = c.resolve("top.S")


def test_recursive_struct_simple():
    with pytest.raises(hwl.DiagnosticException, match="cyclic dependency"):
        c = compile_custom("struct S { a: int, b: S }")
        _ = c.resolve("top.S")


def test_int():
    src = "fn f(x: int(0..8)) {}"
    f: hwl.Function = compile_custom(src).resolve("top.f")

    f(0)
    f(7)
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f(8)
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f(15)


def test_int_multi():
    src = """   
    fn f_basic(x: int(0..8, 20..30)) {}
    fn f_empty(x: int()) {}
    fn f_shuffled(x: int(20..30, 0..8)) {}
    """
    f_basic: hwl.Function = compile_custom(src).resolve("top.f_basic")
    f_empty: hwl.Function = compile_custom(src).resolve("top.f_empty")
    f_shuffled: hwl.Function = compile_custom(src).resolve("top.f_shuffled")

    for f in [f_basic, f_shuffled]:
        f(0)
        f(7)
        with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
            f(8)
        with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
            f(15)
        f(20)
        f(29)
        with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
            f(30)

    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f_empty(0)


def test_int_empty():
    src = "fn f(x: int(0..0)) {}"
    f: hwl.Function = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f(0)


# TODO fix this deadlock by moving all elaboration into a single loop-detecting data structure
@pytest.mark.skip
def test_recursive_struct_generic():
    with pytest.raises(hwl.DiagnosticException, match="cyclic dependency"):
        c = compile_custom("struct S(T: type) { a: int, b: S(T) }")
        s = c.resolve("top.S")
        _ = s(int)
