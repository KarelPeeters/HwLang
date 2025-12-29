from typing import Optional

import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_range_compile():
    check_range_compile(2, "..", 3, valid=True)
    check_range_compile(2, "..", 2, valid=True)
    check_range_compile(2, "..", 1, valid=False)
    check_range_compile(2, "..", 0, valid=False)

    check_range_compile(2, "..=", 3, valid=True)
    check_range_compile(2, "..=", 2, valid=True)
    check_range_compile(2, "..=", 1, valid=True)
    check_range_compile(2, "..=", 0, valid=False)


def test_range_hardware():
    check_range_hardware("0..3", "..", "4..6", valid=True)
    check_range_hardware("0..3", "..", "3..6", valid=True)
    check_range_hardware("0..3", "..", "2..6", valid=True)
    check_range_hardware("0..3", "..", "1..6", valid=False)
    check_range_hardware("0..3", "..", "0..6", valid=False)

    check_range_hardware("0..3", "..=", "4..6", valid=True)
    check_range_hardware("0..3", "..=", "3..6", valid=True)
    check_range_hardware("0..3", "..=", "2..6", valid=True)
    check_range_hardware("0..3", "..=", "1..6", valid=True)
    check_range_hardware("0..3", "..=", "0..6", valid=False)


def check_range_compile(a: Optional[int], ty_range: str, b: Optional[int], valid: bool):
    assert ty_range in ["..", "..=", "..+"]

    src = f"""
    fn f() -> any {{
         return {a if a is not None else ""}{ty_range}{b if b is not None else ""};
    }}
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    if valid:
        f()
    else:
        with pytest.raises(hwl.DiagnosticException, match="range requires"):
            f()


def check_range_hardware(ty_a: Optional[str], ty_range: str, ty_b: Optional[str], valid: bool):
    assert ty_range in ["..", "..=", "..+"]

    src = f"""
    module foo ports(a: in async int({ty_a}), b: in async int({ty_b})) {{
        comb {{
            val r = a{ty_range}b;
        }}
    }}
    """
    c = compile_custom(src)

    if valid:
        _ = c.resolve("top.foo")
    else:
        with pytest.raises(hwl.DiagnosticException, match="range requires"):
            _ = c.resolve("top.foo")
