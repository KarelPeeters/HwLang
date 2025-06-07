import hwl
import pytest

from util import compile_custom


def test_normal_struct():
    c = compile_custom("import std.types.int; struct S { a: int, b: int }")
    _ = c.resolve("top.S")


def test_recursive_struct_simple():
    with pytest.raises(hwl.DiagnosticException, match="cyclic dependency"):
        c = compile_custom("import std.types.int; struct S { a: int, b: S }")
        _ = c.resolve("top.S")


# TODO fix this deadlock by moving all elaboration into a single loop-detecting data structure
@pytest.mark.skip
def test_recursive_struct_generic():
    with pytest.raises(hwl.DiagnosticException, match="cyclic dependency"):
        c = compile_custom("import std.types.int; struct S(T: type) { a: int, b: S(T) }")
        s = c.resolve("top.S")
        _ = s(int)
