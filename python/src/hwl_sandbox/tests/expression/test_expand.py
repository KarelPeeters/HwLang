from pathlib import Path

import pytest

from hwl_sandbox.common.compare import compare_compile


def test_expand_pos(tmp_dir: Path):
    e = compare_compile(["int(0..13)"], "int(0..25)", "return a0;", tmp_dir)
    for v in range(13):
        e.eval_assert([v], v)


def test_expand_neg(tmp_dir: Path):
    e = compare_compile(["int(-24..13)"], "int(-56..25)", "return a0;", tmp_dir)
    for v in range(-24, 13):
        e.eval_assert([v], v)


def test_array(tmp_dir: Path):
    e = compare_compile([], "[4]bool", "return [true, false, true, false];", tmp_dir)
    e.eval_assert([], [True, False, True, False])


@pytest.mark.skip("array indexing does not work yet")
def test_expand_array(tmp_dir: Path):
    # TODO test longer arrays once those work in verilog
    e = compare_compile(["[1]int(4)"], "[1]int(8)", "return a0;", tmp_dir)
    e.eval_assert([[0]], [0])
    e.eval_assert([[1]], [1])
    e.eval_assert([[-1]], [1])
