from pathlib import Path

import pytest

from khdl_sandbox.common.compare import compare_expression
from khdl_sandbox.common.util import compile_custom


# TODO separate syntax for tuple types and values to simplify things
def test_tuple_literal_empty(tmp_dir: Path):
    e = compare_expression([], "Tuple()", "()", tmp_dir)
    e.eval_assert([], ())


def test_tuple_literal_single(tmp_dir: Path):
    e = compare_expression(["uint(8)"], "Tuple(uint(8),)", "(a0,)", tmp_dir)
    e.eval_assert([0], (0,))
    e.eval_assert([16], (16,))


def test_tuple_literal_pair(tmp_dir: Path):
    e = compare_expression(["uint(8)", "bool"], "Tuple(uint(8), bool)", "(a0, a1)", tmp_dir)
    e.eval_assert([0, False], (0, False))
    e.eval_assert([16, True], (16, True))


def test_tuple_extract_single(tmp_dir: Path):
    e = compare_expression(["Tuple(uint(8),)"], "uint(8)", "a0.0", tmp_dir)
    e.eval_assert([(0,)], 0)
    e.eval_assert([(16,)], 16)


@pytest.mark.parametrize("index", [0, 1])
def test_tuple_extract_pair(index: int, tmp_dir: Path):
    e = compare_expression(["Tuple(uint(8), bool)"], ["uint(8)", "bool"][index], f"a0.{index}", tmp_dir)
    e.eval_assert([(0, False)], [0, False][index])
    e.eval_assert([(16, True)], [16, True][index])


def test_tuple_len_compile():
    src = """
    fn f() -> uint { val t = (1, true, 3); return t.len; }
    fn g() -> uint { val t = (); return t.len; }
    fn h() -> uint { return Tuple(bool, uint(8), int).len; }
    fn i() -> uint { return Tuple().len; }
    """
    c = compile_custom(src)
    assert c.resolve("top.f")() == 3
    assert c.resolve("top.g")() == 0
    assert c.resolve("top.h")() == 3
    assert c.resolve("top.i")() == 0


def test_tuple_len_hardware(tmp_dir: Path):
    e = compare_expression(["Tuple(uint(8), bool)"], "uint(8)", "a0.len", tmp_dir)
    e.eval_assert([(0, False)], 2)
