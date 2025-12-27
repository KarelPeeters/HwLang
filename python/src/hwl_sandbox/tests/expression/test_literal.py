from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_expression
from hwl_sandbox.common.util import compile_custom


@pytest.mark.parametrize("v", [False, True])
def test_bool_literal(tmp_dir: Path, v: bool):
    e_true = compare_expression([], "bool", str(v).lower(), tmp_dir)
    e_true.eval_assert([], v)


# include both extremes
@pytest.mark.parametrize("v", [-4, -1, 0, 1, 3])
def test_int_literal(tmp_dir: Path, v: int):
    e_rz = compare_expression([], "int(-4..4)", str(v), tmp_dir)
    e_rz.eval_assert([], v)


def test_int_literal_extra():
    # no need to test the actual simulation here, this is just frontend int parsing
    samples = [
        ("0b0", 0),
        ("0x0", 0),
        ("0b1010101", 0b1010101),
        ("0b101_0101", 0b1010101),

        ("0", 0),
        ("123456789", 123456789),
        ("12345_6789", 123456789),

        ("0x0", 0),
        ("0x0123456789abcdef", 0x0123456789abcdef),
        ("0x0123456789ABCDEF", 0x0123456789abcdef),
        ("0x123_456", 0x123456),
    ]

    for s, i in samples:
        src = f"import std.types.int; fn f() -> int {{ return {s}; }}"
        c = compile_custom(src)
        f: hwl.Function = c.resolve("top.f")
        assert f() == i
