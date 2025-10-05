from pathlib import Path

from hwl_sandbox.common.expression import compile_expression


# TODO test zero-sized everything

def test_expand_pos(tmpdir: Path):
    e = compile_expression(["int(0..13)"], "int(0..25)", "a0", tmpdir)
    for v in range(13):
        e.eval_assert([v], v)


def test_expand_neg(tmpdir: Path):
    e = compile_expression(["int(-24..13)"], "int(-56..25)", "a0", tmpdir)
    for v in range(-24, 13):
        e.eval_assert([v], v)


def test_add_pos(tmpdir: Path):
    e = compile_expression(["int(0..16)", "int(0..32)"], "int(0..48)", "a0 + a1", tmpdir)

    e.eval_assert([0, 0], 0)
    e.eval_assert([0, 1], 1)
    e.eval_assert([15, 31], 46)


def test_negative_simple(tmpdir: Path):
    e = compile_expression(["int(-16..0)", "int(0..=1)"], "int(-16..=0)", "a0 + a1", tmpdir)
    e.eval_assert([-16, 0], -16)


def test_negative_complex(tmpdir: Path):
    e = compile_expression(["int(-902..-640)", "int(-129..-71)"], "int(-4096..=4096)", "a0 + a1", tmpdir)
    e.eval_assert([-644, -91], -735)


def test_negative_result_expanded(tmpdir: Path):
    e = compile_expression(["int(-128..128)", "int(0..128)"], "int(-128..256)", "a0 + a1", tmpdir)
    e.eval_assert([-1, 64], 64 - 1)


def test_negative_result_positive(tmpdir: Path):
    e = compile_expression(["int(-128..128)", "int(128..256)"], "int(-4096..=4096)", "a0 + a1", tmpdir)
    e.eval_assert([-1, 129], 128)
