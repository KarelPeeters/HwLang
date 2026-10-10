from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import CompiledCompare, compare_expression
from hwl_sandbox.common.util import compile_custom, diag_error


def compare_compare(ty_a: str, ty_b: str, tmp_dir: Path, prefix: str = "") -> CompiledCompare:
    return compare_expression([ty_a, ty_b], "Tuple(bool, bool)", "(a0 == a1, a0 != a1)", tmp_dir, prefix=prefix)


def test_compare_bool(tmp_dir: Path):
    e = compare_compare("bool", "bool", tmp_dir)
    for a in [False, True]:
        for b in [False, True]:
            e.eval_assert([a, b], (a == b, a != b))


def test_compare_int(tmp_dir: Path):
    e = compare_compare("int(4)", "uint(4)", tmp_dir)
    for a in range(-2 ** 3, 2 ** 3):
        for b in range(2 ** 4):
            e.eval_assert([a, b], (a == b, a != b))


def test_compare_array(tmp_dir: Path):
    e = compare_compare("[3]bool", "[3]bool", tmp_dir)
    for a in ([False, False, True], [True, False, True]):
        for b in ([False, False, True], [True, False, False]):
            e.eval_assert([a, b], (a == b, a != b))


def test_compare_array_int_different_ranges(tmp_dir: Path):
    e = compare_compare("[2]int(-4..4)", "[2]uint(3)", tmp_dir)
    for a in ([-4, 0], [3, 1]):
        for b in ([3, 1], [0, 3]):
            e.eval_assert([a, b], (a == b, a != b))


def test_compare_array_nested(tmp_dir: Path):
    e = compare_compare("[2][2]bool", "[2][2]bool", tmp_dir)
    for a in ([[False, True], [True, True]], [[False, False], [True, True]]):
        for b in ([[False, True], [True, True]], [[False, True], [True, False]]):
            e.eval_assert([a, b], (a == b, a != b))


def test_compare_array_empty(tmp_dir: Path):
    e = compare_compare("[0]bool", "[0]bool", tmp_dir)
    e.eval_assert([[], []], (True, False))


def test_compare_tuple(tmp_dir: Path):
    e = compare_compare("Tuple(uint(4), bool)", "Tuple(uint(4), bool)", tmp_dir)
    for a in [(3, False), (3, True)]:
        for b in [(3, False), (4, False)]:
            e.eval_assert([a, b], (a == b, a != b))


def test_compare_struct(tmp_dir: Path):
    prefix = """
    struct Pair { x: uint(4), y: bool }
    fn pair(x: uint(4), y: bool) -> Pair { return Pair.new(x=x, y=y); }
    """
    e = compare_compare("Pair", "Pair", tmp_dir, prefix=prefix)
    pair = e.compile.resolve("top.pair")
    for a in [(3, False), (3, True)]:
        for b in [(3, False), (4, False)]:
            e.eval_assert([pair(*a), pair(*b)], (a == b, a != b))


def test_compare_enum_simple(tmp_dir: Path):
    prefix = """
    enum E { A, B, C }
    fn make(i: uint) -> E {
        match (i) {
            0 => { return E.A; }
            1 => { return E.B; }
            2 => { return E.C; }
            _ => { assert(False); }
        }
    }
    """
    e = compare_compare("E", "E", tmp_dir, prefix=prefix)
    make = e.compile.resolve("top.make")
    for a in range(3):
        for b in range(3):
            e.eval_assert([make(a), make(b)], (a == b, a != b))


def test_compare_enum_mixed(tmp_dir: Path):
    prefix = """
    enum E { A, B(uint(4)) }
    fn make(is_b: bool, v: uint(4)) -> E {
        if (is_b) { return E.B(v); } else { return E.A; }
    }
    """
    e = compare_compare("E", "E", tmp_dir, prefix=prefix)
    make = e.compile.resolve("top.make")
    for a in [(False, 0), (False, 3), (True, 3)]:
        for b in [(False, 5), (True, 3), (True, 5)]:
            expected = a[0] == b[0] and (not a[0] or a[1] == b[1])
            e.eval_assert([make(*a), make(*b)], (expected, not expected))


def test_compare_different_types_compile():
    f = compile_custom("fn f(a: any, b: any) -> bool { return a == b; }").resolve("top.f")
    for a, b in [(False, 0), (False, ()), (False, [4])]:
        with pytest.raises(hwl.DiagnosticException):
            f(a, b)


def test_compare_different_types_hardware():
    for a, b in [("bool", "uint(2)"), ("bool", "Tuple(bool)"), ("bool", "[4]uint(2)")]:
        c = compile_custom(
            f"type A = {a}; type B = {b};"
            "module top ports(a: in async A, b: in async B) { comb { a == b; } }"
        )
        with pytest.raises(hwl.DiagnosticException):
            _ = c.resolve("top.top")


def test_compare_compile_values():
    src = """
    struct A {}
    struct B {}
    struct G(b: bool) {}
    fn f(a: any, b: any) -> bool { return a == b; }
    """

    c = compile_custom(src)
    f = c.resolve("top.f")
    a = c.resolve("top.A")
    b = c.resolve("top.B")
    g = c.resolve("top.G")

    assert f("test", "test") is True
    assert f("test", "different") is False

    assert f(a, a) is True
    assert f(a, b) is False

    assert f(g(False), g(False)) is True
    assert f(g(False), g(True)) is False


def test_compare_zero_width_element(tmp_dir: Path):
    e = compare_compare("Tuple(int(3..=3), bool)", "Tuple(int(3..=3), bool)", tmp_dir)
    for a in [False, True]:
        for b in [False, True]:
            e.eval_assert([(3, a), (3, b)], (a == b, a != b))


def test_compare_tuple_mixed(tmp_dir: Path):
    # the compile-time int is outside of the hardware range, so the comparison needs to widen both sides
    e = compare_expression(
        ["Tuple(bool, int(-4..4))", "bool"], "Tuple(bool, bool)", "(a0 == (a1, -1), a0 != (a1, 5))", tmp_dir
    )
    for a in [(False, -1), (True, -1), (True, 3)]:
        for b in [False, True]:
            e.eval_assert([a, b], (a == (b, -1), True))


def test_compare_array_constant_wider(tmp_dir: Path):
    e = compare_expression(["[2]uint(4)"], "Tuple(bool, bool)", "(a0 == [1, 2], a0 != [1, 200])", tmp_dir)
    for a in [[1, 2], [1, 3]]:
        e.eval_assert([a], (a == [1, 2], True))


def test_compare_enum_constant_padding(tmp_dir: Path):
    # variants with smaller payloads are padded, the padding must not influence the comparison
    prefix = """
    enum E { A, B(uint(4)), C(bool) }
    fn make(i: uint(0..3), v: uint(4)) -> E {
        match (i) {
            0 => { return E.A; }
            1 => { return E.B(v); }
            2 => { return E.C(v % 2 == 1); }
        }
    }
    """
    e = compare_expression(
        ["E", "bool"], "Tuple(bool, bool, bool)", "(a0 == E.C(a1), a0 != E.A, a0 == E.B(5))", tmp_dir, prefix=prefix
    )
    make = e.compile.resolve("top.make")
    for i, v in [(0, 0), (1, 5), (1, 3), (2, 0), (2, 1)]:
        for b in [False, True]:
            expected = (i == 2 and (v % 2 == 1) == b, i != 0, i == 1 and v == 5)
            e.eval_assert([make(i, v), b], expected)


def test_compare_errors():
    src = """
    fn f(a: any, b: any) -> bool { return a == b; }
    module top ports(a: in async uint(4)) { comb { "{a}" == "3"; } }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    with diag_error("mismatched operand types for equality operator"):
        f((1,), (1, 2))
    with diag_error("equality between strings or ranges is only supported for compile-time values"):
        c.resolve("top.top")
