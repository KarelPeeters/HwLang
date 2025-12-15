from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_compile
from hwl_sandbox.common.util import compile_custom


def test_bool_to_bits(tmp_dir: Path):
    c = compare_compile(["bool"], "[1]bool", "return bool.to_bits(a0);", tmp_dir)
    c.eval_assert([False], [False])
    c.eval_assert([True], [True])


def test_bool_from_bits(tmp_dir: Path):
    c = compare_compile(["[1]bool"], "bool", "return bool.from_bits(a0);", tmp_dir)
    c.eval_assert([[False]], False)
    c.eval_assert([[True]], True)


def test_int_to_bits_unsigned(tmp_dir: Path):
    c = compare_compile(["int(0..4)"], "[2]bool", "return int(0..4).to_bits(a0);", tmp_dir)
    c.eval_assert([0], [False, False])
    c.eval_assert([1], [True, False])
    c.eval_assert([2], [False, True])
    c.eval_assert([3], [True, True])


def test_bool_from_bits_unsigned(tmp_dir: Path):
    c = compare_compile(["[2]bool"], "int(0..4)", "return int(0..4).from_bits(a0);", tmp_dir)
    c.eval_assert([[False, False]], 0)
    c.eval_assert([[True, False]], 1)
    c.eval_assert([[False, True]], 2)
    c.eval_assert([[True, True]], 3)


def test_int_to_bits_signed(tmp_dir: Path):
    c = compare_compile(["int(-2..2)"], "[2]bool", "return int(-2..2).to_bits(a0);", tmp_dir)
    c.eval_assert([0], [False, False])
    c.eval_assert([1], [True, False])
    c.eval_assert([-2], [False, True])
    c.eval_assert([-1], [True, True])


def test_bool_from_bits_signed(tmp_dir: Path):
    c = compare_compile(["[2]bool"], "int(-2..2)", "return int(-2..2).from_bits(a0);", tmp_dir)
    c.eval_assert([[False, False]], 0)
    c.eval_assert([[True, False]], 1)
    c.eval_assert([[False, True]], -2)
    c.eval_assert([[True, True]], -1)


def test_from_bits_rejected():
    src = """
        import std.types.[bool, int];
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
        import std.types.[bool, int];
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
