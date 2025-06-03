from pathlib import Path

import hwl


def compile_custom(top: str) -> hwl.Compile:
    builder = hwl.SourceBuilder()
    builder.add_tree(["std"], str(Path(__file__).parent / "../../design/project/std"))
    builder.add_file(["top"], "python.top", top)
    return builder.finish().compile()


def test_interact_add():
    c = compile_custom("import std.types.int; fn f(a: int, b: int) -> int { return a + b; }")
    f = c.resolve("top.f")
    assert f(3, 4) == 7


def test_interact_string_array():
    c = compile_custom("import std.types.[int, str]; fn f(a: [3]str, b: int) -> str { return a[b]; }")
    f = c.resolve("top.f")
    assert f(["a", "b", "c"], 1) == "b"
