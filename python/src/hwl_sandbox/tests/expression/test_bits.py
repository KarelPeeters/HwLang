from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_body, compare_expression
from hwl_sandbox.common.util import compile_custom


def test_bool_to_bits(tmp_dir: Path):
    c = compare_expression(["bool"], "[1]bool", "bool.to_bits(a0)", tmp_dir)
    c.eval_assert([False], [False])
    c.eval_assert([True], [True])


def test_bool_from_bits(tmp_dir: Path):
    c = compare_expression(["[1]bool"], "bool", "bool.from_bits(a0)", tmp_dir)
    c.eval_assert([[False]], False)
    c.eval_assert([[True]], True)


def test_verilog_bool_to_bits_index(tmp_dir: Path):
    # edge case:
    #   In verilog we represent bools as single bit values, which don't support immediate indexing.
    #   Here we test that to_bits correctly converts the resulting type to an array of bits.
    c = compare_expression(["bool"], "bool", "bool.to_bits(a0)[0]", tmp_dir)
    c.eval_assert([False], False)
    c.eval_assert([True], True)


def test_int_to_bits_unsigned(tmp_dir: Path):
    c = compare_expression(["int(0..4)"], "[2]bool", "int(0..4).to_bits(a0)", tmp_dir)
    c.eval_assert([0], [False, False])
    c.eval_assert([1], [True, False])
    c.eval_assert([2], [False, True])
    c.eval_assert([3], [True, True])


def test_bool_from_bits_unsigned(tmp_dir: Path):
    c = compare_expression(["[2]bool"], "int(0..4)", "int(0..4).from_bits(a0)", tmp_dir)
    c.eval_assert([[False, False]], 0)
    c.eval_assert([[True, False]], 1)
    c.eval_assert([[False, True]], 2)
    c.eval_assert([[True, True]], 3)


def test_int_to_bits_signed(tmp_dir: Path):
    c = compare_expression(["int(-2..2)"], "[2]bool", "int(-2..2).to_bits(a0)", tmp_dir)
    c.eval_assert([0], [False, False])
    c.eval_assert([1], [True, False])
    c.eval_assert([-2], [False, True])
    c.eval_assert([-1], [True, True])


def test_bool_from_bits_signed(tmp_dir: Path):
    c = compare_expression(["[2]bool"], "int(-2..2)", "int(-2..2).from_bits(a0)", tmp_dir)
    c.eval_assert([[False, False]], 0)
    c.eval_assert([[True, False]], 1)
    c.eval_assert([[False, True]], -2)
    c.eval_assert([[True, True]], -1)


def test_struct_to_bits(tmp_dir: Path):
    prefix = "struct Pair { a: bool, b: uint(0..4) }"
    body = "return Pair.to_bits(Pair.new(a=a0, b=a1));"
    c = compare_body(["bool", "uint(0..4)"], "[3]bool", body, tmp_dir, prefix)
    c.eval_assert([False, 0], [False, False, False])
    c.eval_assert([True, 2], [True, False, True])
    c.eval_assert([False, 3], [False, True, True])


def test_struct_from_bits(tmp_dir: Path):
    prefix = "struct Pair { a: bool, b: uint(0..4) }"
    body = """
    val p = Pair.from_bits(a0);
    return (p.a, p.b);
    """
    c = compare_body(["[3]bool"], "Tuple(bool, uint(0..4))", body, tmp_dir, prefix)
    c.eval_assert([[False, False, False]], (False, 0))
    c.eval_assert([[True, False, True]], (True, 2))
    c.eval_assert([[False, True, True]], (False, 3))


def test_enum_to_bits(tmp_dir: Path):
    prefix = "enum E { Empty, Left(bool), Right(uint(0..4)) }"
    body = """
    val e: E;
    if (a0 == 0) {
        e = E.Empty;
    } else if (a0 == 1) {
        e = E.Left(a1);
    } else {
        e = E.Right(a2);
    }
    return E.to_bits(e);
    """
    c = compare_body(["int(0..3)", "bool", "uint(0..4)"], "[4]bool", body, tmp_dir, prefix)
    c.eval_assert([0, False, 0], [False, False, False, False])
    c.eval_assert([1, False, 0], [True, False, False, False])
    c.eval_assert([1, True, 0], [True, False, True, False])
    c.eval_assert([2, False, 2], [False, True, False, True])


def test_enum_from_bits(tmp_dir: Path):
    prefix = "enum E { Empty, Left(bool), Right(uint(0..4)) }"
    body = """
    val e = E.from_bits_unsafe(a0);
    match (e) {
        .Empty => { return (0, false, 0); }
        .Left(val v) => { return (1, v, 0); }
        .Right(val v) => { return (2, false, v); }
    }
    """
    c = compare_body(["[4]bool"], "Tuple(uint(0..3), bool, uint(0..4))", body, tmp_dir, prefix)
    c.eval_assert([[False, False, False, False]], (0, False, 0))
    c.eval_assert([[True, False, False, False]], (1, False, 0))
    c.eval_assert([[True, False, True, False]], (1, True, 0))
    c.eval_assert([[False, True, False, True]], (2, False, 2))


def test_from_bits_rejected():
    src = """
        fn f(x: [2]bool) -> int(0..3) {
            return int(0..3).from_bits(x);
        }
    """
    c = compile_custom(src)
    f: hwl.Function = c.resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="only allowed for types where every bit pattern is valid"):
        f([False, False])


def test_from_bits_unsafe_compile():
    src = """
        fn f(x: [2]bool) -> int(0..3) {
            return int(0..3).from_bits_unsafe(x);
        }
    """
    c = compile_custom(src)
    f: hwl.Function = c.resolve("top.f")

    assert f([False, False]) == 0
    assert f([True, False]) == 1
    assert f([False, True]) == 2
    with pytest.raises(hwl.DiagnosticException, match="invalid bit pattern"):
        f([True, True])


def test_size_bits():
    src = """
    fn f(T: type) -> int { return T.size_bits; }
    fn ty_array(N: uint, T: type) -> type { return [N]T; }  
    struct Pair(A: type, B: type) { a: A, b: B } 
    """
    c = compile_custom(src)
    f: hwl.Function = c.resolve("top.f")

    c_uint = c.resolve("std.types.uint")
    c_int = c.resolve("std.types.int")
    c_tuple = c.resolve("std.types.Tuple")
    c_array = c.resolve("top.ty_array")
    c_pair = c.resolve("top.Pair")

    assert f(bool) == 1
    assert f(c_uint(0)) == 0
    assert f(c_uint(4)) == 4
    assert f(c_int(4)) == 4
    assert f(c_array(0, bool)) == 0
    assert f(c_array(4, c_uint(0))) == 0
    assert f(c_array(4, c_uint(3))) == 12
    assert f(c_pair(bool, c_uint(3))) == 4
    assert f(c_tuple(bool, c_uint(3))) == 4
    assert f(c_tuple()) == 0
