from pathlib import Path

import pytest

from hwl_sandbox.common.compare import compare_expression
from hwl_sandbox.common.util import compile_custom


@pytest.mark.parametrize("v", [False, True])
def test_literal_bool(tmp_dir: Path, v: bool):
    e = compare_expression([], "bool", str(v).lower(), tmp_dir)
    e.eval_assert([], v)


# include both extremes
@pytest.mark.parametrize("v", [-4, -1, 0, 1, 3])
def test_literal_int_basic(tmp_dir: Path, v: int):
    e = compare_expression([], "int(-4..4)", str(v), tmp_dir)
    e.eval_assert([], v)


def test_literal_int_extra():
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

    for source, expected in samples:
        src = f"const cst = {source};"
        cst = compile_custom(src).resolve("top.cst")
        assert cst == expected


def test_literal_string():
    src = """
    const cst = [
        "",
        "test",
        "{4} {5}",
        r"{4} {5}",
        "\\{4} {5}",
        "\\\\",
        "\\"",
        "\\n",
        "\\r",
        "\\t",
        "\\0",
    ];
    """
    expected = [
        "",
        "test",
        "4 5",
        "{4} {5}",
        "{4} 5",
        "\\",
        "\"",
        "\n",
        "\r",
        "\t",
        "\0",
    ]

    cst = compile_custom(src).resolve("top.cst")
    assert cst == expected
