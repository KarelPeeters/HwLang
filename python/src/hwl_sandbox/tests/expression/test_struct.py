from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_body
from hwl_sandbox.common.util import compile_custom


def test_type_normal_struct():
    c = compile_custom("struct S { a: int, b: int }")
    _ = c.resolve("top.S")


def test_type_recursive_struct_simple():
    with pytest.raises(hwl.DiagnosticException, match="cyclic dependency"):
        c = compile_custom("struct S { a: int, b: S }")
        _ = c.resolve("top.S")


def test_struct_simple_basics(tmp_dir: Path):
    prefix = """
    struct Pair { x: uint(8), y: bool }
    """
    src = """
    val v = Pair.new(x=a0, y=a1);
    return (v.y, v.x);
    """

    e = compare_body(["uint(8)", "bool"], "Tuple(bool, uint(8))", src, tmp_dir, prefix=prefix)

    e.eval_assert([0, False], (False, 0))
    e.eval_assert([5, False], (False, 5))
    e.eval_assert([0, True], (True, 0))


def test_struct_generic_basics(tmp_dir: Path):
    prefix = """
    struct Pair(T: type) { x: uint(8), y: T }
    """
    src = """
    val v = Pair(bool).new(x=a0, y=a1);
    return (v.y, v.x);
    """

    e = compare_body(["uint(8)", "bool"], "Tuple(bool, uint(8))", src, tmp_dir, prefix=prefix)

    e.eval_assert([0, False], (False, 0))
    e.eval_assert([5, False], (False, 5))
    e.eval_assert([0, True], (True, 0))
