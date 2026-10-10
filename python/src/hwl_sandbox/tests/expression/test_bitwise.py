from pathlib import Path

import pytest
from typing import Tuple, Callable, Any

from hwl_sandbox.common.compare import compare_expression
from hwl_sandbox.common.util import compile_custom, diag_error

OP = Callable[[Any, Any], Any]
OPS_BITWISE = [
    pytest.param(("&", lambda a, b: a & b), id="and"),
    pytest.param(("|", lambda a, b: a | b), id="or"),
    pytest.param(("^", lambda a, b: a ^ b), id="xor"),
]


@pytest.mark.parametrize("op", OPS_BITWISE)
def test_bitwise_int(op: Tuple[str, OP], tmp_dir: Path):
    op_str, op_py = op
    e = compare_expression(["int(-8..8)", "int(0..16)"], "int(-16..16)", f"a0 {op_str} a1", tmp_dir)
    for a in [-8, -5, -1, 0, 3, 7]:
        for b in [0, 1, 6, 15]:
            e.eval_assert([a, b], op_py(a, b))


@pytest.mark.parametrize("op", OPS_BITWISE)
def test_bitwise_bool_scalar(op: Tuple[str, OP], tmp_dir: Path):
    op_str, op_py = op
    e = compare_expression(["bool", "bool"], "bool", f"a0 {op_str} a1", tmp_dir)
    for a in [False, True]:
        for b in [False, True]:
            e.eval_assert([a, b], op_py(a, b))


@pytest.mark.parametrize("op", OPS_BITWISE)
def test_bitwise_bool_scalar_array(op: Tuple[str, OP], tmp_dir: Path):
    op_str, op_py = op
    e = compare_expression(["bool", "[3]bool"], "[3]bool", f"a0 {op_str} a1", tmp_dir)
    for a in [False, True]:
        for b in ([True, False, True], [True, True, False]):
            e.eval_assert([a, b], [op_py(a, y) for y in b])


@pytest.mark.parametrize("op", OPS_BITWISE)
def test_bitwise_bool_array_array(op: Tuple[str, OP], tmp_dir: Path):
    op_str, op_py = op
    e = compare_expression(["[3]bool", "[3]bool"], "[3]bool", f"a0 {op_str} a1", tmp_dir)
    for a in ([False, False, True], [False, True, True]):
        for b in ([True, False, True], [True, True, False]):
            e.eval_assert([a, b], [op_py(x, y) for x, y in zip(a, b)])


def test_bitwise_array_length_mismatch():
    f = compile_custom("fn f(a: [3]bool, b: [4]bool) -> any { return a & b; }").resolve("top.f")
    with diag_error("bitwise operator on arrays with different lengths"):
        f([False] * 3, [False] * 4)


def test_bitwise_not_int(tmp_dir: Path):
    e = compare_expression(["int(-4..16)"], "int(-16..4)", "!a0", tmp_dir)
    for i in range(-4, 16):
        e.eval_assert([i], ~i)


def test_bitwise_not_array(tmp_dir: Path):
    e = compare_expression(["[3]bool"], "[3]bool", "!a0", tmp_dir)
    e.eval_assert([[False, True, True]], [True, False, False])
