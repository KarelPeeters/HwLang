from pathlib import Path

import pytest

from hwl_sandbox.common.compare import compare_expression


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
