import pytest

from hwl_sandbox.common.util import compile_custom, diag_error


def test_type_int_simple():
    src = "fn f(x: int(0..8)) {}"
    f = compile_custom(src).resolve("top.f")

    f(0)
    f(7)
    with diag_error("type mismatch"):
        f(8)
    with diag_error("type mismatch"):
        f(15)


def test_type_int_multi():
    src = """   
    fn f_basic(x: int(0..8, 20..30)) {}
    fn f_empty(x: int()) {}
    fn f_shuffled(x: int(20..30, 0..8)) {}
    """
    f_basic = compile_custom(src).resolve("top.f_basic")
    f_empty = compile_custom(src).resolve("top.f_empty")
    f_shuffled = compile_custom(src).resolve("top.f_shuffled")

    for f in [f_basic, f_shuffled]:
        f(0)
        f(7)
        with diag_error("type mismatch"):
            f(8)
        with diag_error("type mismatch"):
            f(15)
        f(20)
        f(29)
        with diag_error("type mismatch"):
            f(30)

    with diag_error("type mismatch"):
        f_empty(0)


def test_type_int_empty():
    src = "fn f(x: int(0..0)) {}"
    f = compile_custom(src).resolve("top.f")
    with diag_error("type mismatch"):
        f(0)


# TODO fix this deadlock by moving all elaboration into a single loop-detecting data structure
@pytest.mark.skip
def test_type_recursive_struct_generic():
    with diag_error("cyclic dependency"):
        c = compile_custom("struct S(T: type) { a: int, b: S(T) }")
        s = c.resolve("top.S")
        _ = s(int)


def test_type_array():
    src = """
    fn f(x: [_]bool) -> uint { return x.len; }
    fn g(x: [2]bool) -> uint { return x.len; }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    g = c.resolve("top.g")

    assert f([]) == 0
    assert f([True]) == 1
    assert f([True, False]) == 2

    with diag_error("type mismatch"):
        assert f([0]) == 2

    assert g([True, False]) == 2
    with diag_error("type mismatch"):
        assert g([True]) == 1
    with diag_error("type mismatch"):
        assert g([0, 1]) == 3


def test_type_tuple_index():
    src = """
    fn f() -> type {
        type T = Tuple(bool, uint);
        val v: T.1 = 4;
        return T.0;
    }
    fn g() -> type {
        type T = Tuple;
        return T.0;
    }
    """

    f = compile_custom(src).resolve("top.f")
    g = compile_custom(src).resolve("top.g")

    assert f() == bool
    with diag_error("cannot index into tuple type with unknown field types"):
        g()
