from pathlib import Path

import hwl
import pytest
from hwl.hwl import DiagnosticException

from hwl_sandbox.common.util import compile_custom


def test_interact_add():
    c = compile_custom("fn f(a: int, b: int) -> int { return a + b; }")
    f = c.resolve("top.f")
    assert f(3, 4) == 7


def test_interact_string_array():
    c = compile_custom("fn f(a: [3]str, b: int) -> str { return a[b]; }")
    f = c.resolve("top.f")
    assert f(["a", "b", "c"], 1) == "b"


def test_interact_types():
    c = compile_custom("fn f(T: type, x: T) -> T { return x; }")
    f = c.resolve("top.f")

    assert f(bool, False) is False
    assert f(bool, True) is True
    with pytest.raises(DiagnosticException, match="type mismatch"):
        f(bool, 0)

    assert f(int, 0) == 0
    with pytest.raises(DiagnosticException, match="type mismatch"):
        f(int, False)


def test_compile_manifest():
    manifest_path = Path(__file__).parent / "project/hwl.toml"
    s = hwl.Source.new_from_manifest_path(str(manifest_path))
    c = s.compile()
    assert isinstance(c.resolve("top.top"), hwl.Module)


def test_format():
    src = "const c = a+b;"
    expected = "const c = a + b;\n"
    assert hwl.format_file(src) == expected


def test_capture_prints():
    src = """
    fn f(a: int) -> int {
        print("hello");
        print("world");
        return a + 1;
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")

    with c.capture_prints() as capture:
        result = f(5)

    assert result == 6
    assert capture.prints == ["hello\n", "world\n"]


def test_call_type():
    c = compile_custom("")
    uint = c.resolve("std.types.uint")
    assert str(uint) == "uint"
    assert str(uint(8)) == "int(0..256)"


def test_interact_struct(tmp_dir: Path):
    src = """
    struct Pair { x: uint(8), y: bool }
    fn f(x: uint(8), y: bool) -> Pair {
        return Pair.new(x=x, y=y);
    }
    """

    c = compile_custom(src)
    pair = c.resolve("top.Pair")
    f = c.resolve("top.f")

    # check struct construction, indirectly and directly
    assert str(f(4, False)) == "Pair.new(x=4, y=false)"
    assert str(pair.new(x=4, y=False)) == "Pair.new(x=4, y=false)"

    # check struct equality
    a0 = f(4, False)
    a1 = f(4, False)
    b = f(5, False)
    assert a0 == a0
    assert a0 == a1
    assert not (a0 == b)
    assert not (a0 != a1)
    assert a0 != b

    # check comparing values python values works as expected
    assert not (a0 == "test")
    assert a0 != "test"

    # check that we get normal python behavior for non-existing attributes
    with pytest.raises(AttributeError):
        _ = f(4, False).non_existing


def test_interact_enum():
    src = """
    enum Foo { Empty, Data(uint(8)) }
    fn f(x: bool, y: uint(8)) -> Foo {
        if (x) {
            return Foo.Data(y);
        } else {
            return Foo.Empty;
        }
    }
    """

    c = compile_custom(src)
    foo = c.resolve("top.Foo")
    f = c.resolve("top.f")

    # check enum construction, indirectly and directly
    assert str(f(False, 0)) == "Foo.Empty"
    assert str(f(True, 0)) == "Foo.Data(0)"
    assert str(f(True, 1)) == "Foo.Data(1)"

    assert str(foo.Empty) == "Foo.Empty"
    assert str(foo.Data(0)) == "Foo.Data(0)"
    assert str(foo.Data(1)) == "Foo.Data(1)"
