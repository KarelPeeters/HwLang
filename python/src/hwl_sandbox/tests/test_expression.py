from pathlib import Path

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


def assert_div(e: CompiledExpression, a: int, b: int, c: int):
    assert a // b == c
    e.eval_assert([a, b], c)


def test_div_simple(tmpdir: Path):
    e = expression_compile(["int(0..16)", "int(1..32)"], "int(0..16)", "a0 / a1", tmpdir)
    assert_div(e, 15, 3, 5)
    assert_div(e, 15, 1, 15)
    assert_div(e, 15, 30, 0)


# TODO bruteforce loop over all possible ranges and then all possible values
def test_div_by_pos(tmpdir: Path):
    e = expression_compile(["int(-16..16)", "int(1..16)"], "int(-16..16)", "a0 / a1", tmpdir)
    assert_div(e, 15, 3, 5)
    assert_div(e, -15, 3, -5)
    assert_div(e, 15, 2, 7)
    assert_div(e, -15, 2, -8)


def test_div_by_neg(tmpdir: Path):
    e = expression_compile(["int(-16..16)", "int(-16..0)"], "int(-16..=16)", "a0 / a1", tmpdir)
    assert_div(e, 15, -3, -5)
    assert_div(e, -15, -3, 5)
    assert_div(e, 15, -2, -8)
    assert_div(e, -15, -2, 7)


def test_div_result_neg(tmpdir: Path):
    e = expression_compile(["int(-16..0)", "int(1..16)"], "int(-16..0)", "a0 / a1", tmpdir)
    assert_div(e, -15, 3, -5)
    assert_div(e, -15, 2, -8)


def test_div_result_pos(tmpdir: Path):
    e = expression_compile(["int(-16..0)", "int(-16..=-1)"], "int(0..=16)", "a0 / a1", tmpdir)
    assert_div(e, -15, -3, 5)
    assert_div(e, -15, -2, 7)


def test_div_tricky(tmpdir: Path):
    e = expression_compile(["int(0..2107)", "int(-8..=-5)"], "int(-59332..=222)", "a0 / a1", tmpdir)
    assert_div(e, 2106, -6, -351)
