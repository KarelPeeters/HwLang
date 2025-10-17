from pathlib import Path

import pytest

from hwl_sandbox.common.expression import expression_compile, CompiledExpression


def test_expand_pos(tmpdir: Path):
    e = expression_compile(["int(0..13)"], "int(0..25)", "a0", tmpdir)
    for v in range(13):
        e.eval_assert([v], v)


def test_expand_neg(tmpdir: Path):
    e = expression_compile(["int(-24..13)"], "int(-56..25)", "a0", tmpdir)
    for v in range(-24, 13):
        e.eval_assert([v], v)


def test_add_pos(tmpdir: Path):
    e = expression_compile(["int(0..16)", "int(0..32)"], "int(0..48)", "a0 + a1", tmpdir)

    e.eval_assert([0, 0], 0)
    e.eval_assert([0, 1], 1)
    e.eval_assert([15, 31], 46)


def test_negative_simple(tmpdir: Path):
    e = expression_compile(["int(-16..0)", "int(0..=1)"], "int(-16..=0)", "a0 + a1", tmpdir)
    e.eval_assert([-16, 0], -16)


def test_negative_complex(tmpdir: Path):
    e = expression_compile(["int(-902..-640)", "int(-129..-71)"], "int(-4096..=4096)", "a0 + a1", tmpdir)
    e.eval_assert([-644, -91], -735)


def test_negative_result_expanded(tmpdir: Path):
    e = expression_compile(["int(-128..128)", "int(0..128)"], "int(-128..256)", "a0 + a1", tmpdir)
    e.eval_assert([-1, 64], 64 - 1)


def test_negative_result_positive(tmpdir: Path):
    e = expression_compile(["int(-128..128)", "int(128..256)"], "int(-4096..=4096)", "a0 + a1", tmpdir)
    e.eval_assert([-1, 129], 128)


def test_zero_width(tmpdir: Path):
    e = expression_compile(["int(0..=0)", "int(0..16)"], "int(0..16)", "a0 + a1", tmpdir)
    e.eval_assert([0, 8], 8)


def test_constant_non_zero(tmpdir: Path):
    e = expression_compile(["int(5..=5)", "int(0..16)"], "int(5..21)", "a0 + a1", tmpdir)
    e.eval_assert([5, 8], 13)


def test_zero_width_outside_result_range(tmpdir: Path):
    e = expression_compile(["int(-1..0)", "int(0..1)"], "int(-8..8)", "a0 + a1", tmpdir)
    e.eval_assert([-1, 0], -1)


def test_large_port(tmpdir: Path):
    e = expression_compile(["int(0..2**128)", "int(0..=0)"], "int(0..2**128)", "a0 + a1", tmpdir)
    e.eval_assert([0, 0], 0)
    e.eval_assert([2 ** 63 - 1, 0], 2 ** 63 - 1)
    e.eval_assert([2 ** 64 - 1, 0], 2 ** 64 - 1)
    e.eval_assert([2 ** 65 - 1, 0], 2 ** 65 - 1)
    e.eval_assert([2 ** 128 - 1, 0], 2 ** 128 - 1)


def assert_div_or_mod(e: CompiledExpression, op: str, a: int, b: int, c_div: int, c_mod: int):
    assert a // b == c_div
    assert a % b == c_mod

    if op == "/":
        e.eval_assert([a, b], c_div)
    elif op == "%":
        e.eval_assert([a, b], c_mod)
    else:
        raise ValueError(f"Invalid op {op}")


OPS_DIV_MOD = ["/", "%"]


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_simple(tmpdir: Path, op: str):
    e = expression_compile(["int(0..16)", "int(1..32)"], "int(0..32)", f"a0 {op} a1", tmpdir)
    assert_div_or_mod(e, op, 15, 3, 5, 0)
    assert_div_or_mod(e, op, 15, 1, 15, 0)
    assert_div_or_mod(e, op, 15, 30, 0, 15)


# TODO bruteforce loop over all possible ranges and then all possible values
@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_by_pos(tmpdir: Path, op: str):
    e = expression_compile(["int(-16..16)", "int(1..16)"], "int(-16..16)", f"a0 {op} a1", tmpdir)
    assert_div_or_mod(e, op, 15, 3, 5, 0)
    assert_div_or_mod(e, op, -15, 3, -5, 0)
    assert_div_or_mod(e, op, 15, 2, 7, 1)
    assert_div_or_mod(e, op, -15, 2, -8, 1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_by_neg(tmpdir: Path, op: str):
    e = expression_compile(["int(-16..16)", "int(-16..0)"], "int(-16..=16)", f"a0 {op} a1", tmpdir)
    assert_div_or_mod(e, op, 15, -3, -5, 0)
    assert_div_or_mod(e, op, -15, -3, 5, 0)
    assert_div_or_mod(e, op, 15, -2, -8, -1)
    assert_div_or_mod(e, op, -15, -2, 7, -1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_result_neg(tmpdir: Path, op: str):
    result_type = "int(-16..0)" if op == "/" else "int(0..15)"
    e = expression_compile(["int(-16..0)", "int(1..16)"], result_type, f"a0 {op} a1", tmpdir)
    assert_div_or_mod(e, op, -15, 3, -5, 0)
    assert_div_or_mod(e, op, -15, 2, -8, 1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_result_pos(tmpdir: Path, op: str):
    result_type = "int(0..=16)" if op == "/" else "int(-15..=0)"
    e = expression_compile(["int(-16..0)", "int(-16..=-1)"], result_type, f"a0 {op} a1", tmpdir)
    assert_div_or_mod(e, op, -15, -3, 5, 0)
    assert_div_or_mod(e, op, -15, -2, 7, -1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_tricky(tmpdir: Path, op: str):
    e = expression_compile(["int(0..2107)", "int(-8..=-5)"], "int(-59332..=222)", f"a0 {op} a1", tmpdir)
    assert_div_or_mod(e, op, 2106, -6, -351, 0)


def test_compare_signed(tmpdir: Path):
    e = expression_compile(["int(-16..16)", "int(-16..16)"], "bool", "a0 < a1", tmpdir)
    e.eval_assert([-1, 1], True)


def test_bool_literals(tmpdir: Path):
    e_true = expression_compile([], "bool", "true", tmpdir)
    e_true.eval_assert([], True)
    
    e_false = expression_compile([], "bool", "false", tmpdir)
    e_false.eval_assert([], False)
