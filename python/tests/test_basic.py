from util import compile_custom


def test_interact_add():
    c = compile_custom("import std.types.int; fn f(a: int, b: int) -> int { return a + b; }")
    f = c.resolve("top.f")
    assert f(3, 4) == 7


def test_interact_string_array():
    c = compile_custom("import std.types.[int, str]; fn f(a: [3]str, b: int) -> str { return a[b]; }")
    f = c.resolve("top.f")
    assert f(["a", "b", "c"], 1) == "b"
