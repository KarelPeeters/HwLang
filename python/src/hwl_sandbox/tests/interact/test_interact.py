from pathlib import Path

import hwl
import pytest
from hwl.hwl import DiagnosticException

from hwl_sandbox.common.util import compile_custom


def test_interact_add():
    c = compile_custom("import std.types.int; fn f(a: int, b: int) -> int { return a + b; }")
    f = c.resolve("top.f")
    assert f(3, 4) == 7


def test_interact_string_array():
    c = compile_custom("import std.types.[int, str]; fn f(a: [3]str, b: int) -> str { return a[b]; }")
    f = c.resolve("top.f")
    assert f(["a", "b", "c"], 1) == "b"


def test_interact_types():
    c = compile_custom("import std.types.bool; fn f(T: type, x: T) -> T { return x; }")
    f = c.resolve("top.f")

    assert f(bool, False) is False
    assert f(bool, True) is True
    with pytest.raises(DiagnosticException, match="type mismatch"):
        f(bool, 0)

    assert f(int, 0) == 0
    with pytest.raises(DiagnosticException, match="type mismatch"):
        f(int, False)


# TODO this is a bit of a weird test, since the example project changes a lot
def test_compile_manifest():
    manifest_path = Path(__file__).parent / "../../../../../design/project/hwl.toml"
    s = hwl.Source.new_from_manifest_path(str(manifest_path))
    c = s.compile()
    assert isinstance(c.resolve("top.top"), hwl.Module)


def test_format():
    src = "const c = a+b;"
    expected = "const c = a + b;\n"
    assert hwl.format_file(src) == expected


def test_capture_prints():
    c = compile_custom(
        """
        import std.types.int;
        import std.util.print;
        fn f(a: int) -> int {
            print("hello");
            print("world");
            return a + 1;
        }
        """
    )
    f = c.resolve("top.f")

    with c.capture_prints() as capture:
        result = f(5)

    assert result == 6
    assert capture.prints == ["hello\n", "world\n"]
