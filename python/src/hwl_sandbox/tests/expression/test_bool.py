from pathlib import Path
from hwl_sandbox.common.compare import compare_expression

BOOL_PAIRS = [(False, False), (False, True), (True, False), (True, True)]


# TODO test short circuiting once that's implemented

def test_bool_and(tmp_dir: Path):
    e = compare_expression(["bool", "bool"], "bool", "a0 && a1", tmp_dir)
    for a, b in BOOL_PAIRS:
        e.eval_assert([a, b], a and b)


def test_bool_or(tmp_dir: Path):
    e = compare_expression(["bool", "bool"], "bool", "a0 || a1", tmp_dir)
    for a, b in BOOL_PAIRS:
        e.eval_assert([a, b], a or b)


def test_bool_xor(tmp_dir: Path):
    e = compare_expression(["bool", "bool"], "bool", "a0 ^^ a1", tmp_dir)
    for a, b in BOOL_PAIRS:
        e.eval_assert([a, b], a ^ b)


def test_bool_bit_and(tmp_dir: Path):
    e = compare_expression(["bool", "bool"], "bool", "a0 & a1", tmp_dir)
    for a, b in BOOL_PAIRS:
        e.eval_assert([a, b], a and b)


def test_bool_bit_or(tmp_dir: Path):
    e = compare_expression(["bool", "bool"], "bool", "a0 | a1", tmp_dir)
    for a, b in BOOL_PAIRS:
        e.eval_assert([a, b], a or b)


def test_bool_bit_xor(tmp_dir: Path):
    e = compare_expression(["bool", "bool"], "bool", "a0 ^ a1", tmp_dir)
    for a, b in BOOL_PAIRS:
        e.eval_assert([a, b], a ^ b)
