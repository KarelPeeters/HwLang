from pathlib import Path

import pytest

from hwl_sandbox.common.compare import CompiledCompare, compare_expression


def test_add_pos(tmp_dir: Path):
    e = compare_expression(["int(0..16)", "int(0..32)"], "int(0..48)", "a0 + a1", tmp_dir)

    e.eval_assert([0, 0], 0)
    e.eval_assert([0, 1], 1)
    e.eval_assert([15, 31], 46)


def test_add_negative_simple(tmp_dir: Path):
    e = compare_expression(["int(-16..0)", "int(0..=1)"], "int(-16..=0)", "a0 + a1", tmp_dir)
    e.eval_assert([-16, 0], -16)


def test_add_negative_complex(tmp_dir: Path):
    e = compare_expression(["int(-902..-640)", "int(-129..-71)"], "int(-4096..=4096)", "a0 + a1", tmp_dir)
    e.eval_assert([-644, -91], -735)


def test_add_negative_result_expanded(tmp_dir: Path):
    e = compare_expression(["int(-128..128)", "int(0..128)"], "int(-128..256)", "a0 + a1", tmp_dir)
    e.eval_assert([-1, 64], 64 - 1)


def test_add_negative_result_positive(tmp_dir: Path):
    e = compare_expression(["int(-128..128)", "int(128..256)"], "int(-4096..=4096)", "a0 + a1", tmp_dir)
    e.eval_assert([-1, 129], 128)


def test_add_zero_width(tmp_dir: Path):
    e = compare_expression(["int(0..=0)", "int(0..16)"], "int(0..16)", "a0 + a1", tmp_dir)
    e.eval_assert([0, 8], 8)


def test_add_constant_non_zero(tmp_dir: Path):
    e = compare_expression(["int(5..=5)", "int(0..16)"], "int(5..21)", "a0 + a1", tmp_dir)
    e.eval_assert([5, 8], 13)


def test_add_zero_width_outside_result_range(tmp_dir: Path):
    e = compare_expression(["int(-1..0)", "int(0..1)"], "int(-8..8)", "a0 + a1", tmp_dir)
    e.eval_assert([-1, 0], -1)


def test_add_large_port(tmp_dir: Path):
    e = compare_expression(["int(0..2**128)", "int(0..=0)"], "int(0..2**128)", "a0 + a1", tmp_dir)
    e.eval_assert([0, 0], 0)
    e.eval_assert([2 ** 63 - 1, 0], 2 ** 63 - 1)
    e.eval_assert([2 ** 64 - 1, 0], 2 ** 64 - 1)
    e.eval_assert([2 ** 65 - 1, 0], 2 ** 65 - 1)
    e.eval_assert([2 ** 128 - 1, 0], 2 ** 128 - 1)


def test_unary_neg(tmp_dir: Path):
    e = compare_expression(["int(-4..16)"], "int(-15..5)", "-a0", tmp_dir)
    for i in range(-4, 16):
        e.eval_assert([i], -i)


def test_unary_plus(tmp_dir: Path):
    e = compare_expression(["int(-4..16)"], "int(-4..16)", "+a0", tmp_dir)
    for i in range(-4, 16):
        e.eval_assert([i], i)


def assert_div_or_mod(e: CompiledCompare, op: str, a: int, b: int, c_div: int, c_mod: int):
    assert a // b == c_div
    assert a % b == c_mod

    if op == "/":
        e.eval_assert([a, b], c_div)
    elif op == "%":
        e.eval_assert([a, b], c_mod)
    else:
        raise ValueError(f"Invalid op {op}")


# TODO instead of duplicating tests, just return a tuple
OPS_DIV_MOD = ["/", "%"]


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_simple(tmp_dir: Path, op: str):
    e = compare_expression(["int(0..16)", "int(1..32)"], "int(0..32)", f"a0 {op} a1", tmp_dir)
    assert_div_or_mod(e, op, 15, 3, 5, 0)
    assert_div_or_mod(e, op, 15, 1, 15, 0)
    assert_div_or_mod(e, op, 15, 30, 0, 15)


# TODO bruteforce loop over all possible ranges and then all possible values
@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_by_pos(tmp_dir: Path, op: str):
    e = compare_expression(["int(-16..16)", "int(1..16)"], "int(-16..16)", f"a0 {op} a1", tmp_dir)
    assert_div_or_mod(e, op, 15, 3, 5, 0)
    assert_div_or_mod(e, op, -15, 3, -5, 0)
    assert_div_or_mod(e, op, 15, 2, 7, 1)
    assert_div_or_mod(e, op, -15, 2, -8, 1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_by_neg(tmp_dir: Path, op: str):
    e = compare_expression(["int(-16..16)", "int(-16..0)"], "int(-16..=16)", f"a0 {op} a1", tmp_dir)
    assert_div_or_mod(e, op, 15, -3, -5, 0)
    assert_div_or_mod(e, op, -15, -3, 5, 0)
    assert_div_or_mod(e, op, 15, -2, -8, -1)
    assert_div_or_mod(e, op, -15, -2, 7, -1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_result_neg(tmp_dir: Path, op: str):
    result_type = "int(-16..0)" if op == "/" else "int(0..15)"
    e = compare_expression(["int(-16..0)", "int(1..16)"], result_type, f"a0 {op} a1", tmp_dir)
    assert_div_or_mod(e, op, -15, 3, -5, 0)
    assert_div_or_mod(e, op, -15, 2, -8, 1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_result_pos(tmp_dir: Path, op: str):
    result_type = "int(0..=16)" if op == "/" else "int(-15..=0)"
    e = compare_expression(["int(-16..0)", "int(-16..=-1)"], result_type, f"a0 {op} a1", tmp_dir)
    assert_div_or_mod(e, op, -15, -3, 5, 0)
    assert_div_or_mod(e, op, -15, -2, 7, -1)


@pytest.mark.parametrize("op", OPS_DIV_MOD)
def test_div_tricky(tmp_dir: Path, op: str):
    e = compare_expression(["int(0..2107)", "int(-8..=-5)"], "int(-59332..=222)", f"a0 {op} a1", tmp_dir)
    assert_div_or_mod(e, op, 2106, -6, -351, 0)


def test_compare_signed(tmp_dir: Path):
    e = compare_expression(["int(-16..16)", "int(-16..16)"], "bool", "a0 < a1", tmp_dir)
    e.eval_assert([-1, 1], True)


def test_shift_left_basic(tmp_dir: Path):
    e = compare_expression(["int(0..16)", "int(0..4)"], "int(0..256)", "a0 << a1", tmp_dir)
    for a in [0, 3, 15]:
        for i in range(4):
            e.eval_assert([a, i], a << i)


def test_shift_right_basic(tmp_dir: Path):
    e = compare_expression(["int(0..16)", "int(0..4)"], "int(0..256)", "a0 >> a1", tmp_dir)
    for a in [0, 3, 15]:
        for i in range(4):
            e.eval_assert([a, i], a >> i)
