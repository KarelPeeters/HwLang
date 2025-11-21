from pathlib import Path

from hwl import hwl

from hwl_sandbox.common.util import compile_custom


def test_interact_add():
    c = compile_custom("import std.types.int; fn f(a: int, b: int) -> int { return a + b; }")
    f = c.resolve("top.f")
    assert f(3, 4) == 7


def test_interact_string_array():
    c = compile_custom("import std.types.[int, str]; fn f(a: [3]str, b: int) -> str { return a[b]; }")
    f = c.resolve("top.f")
    assert f(["a", "b", "c"], 1) == "b"


def test_compile_manifest():
    manifest_path = Path(__file__).parent / "../../../../design/project/hwl.toml"
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
    assert capture.prints == ["hello", "world"]
