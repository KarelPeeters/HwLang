from pathlib import Path

from hwl_sandbox.common.compare import compare_expression, compare_body

PREFIX = """
struct Foo {
    a: bool,
    b: Tuple(bool, [4]uint(8), bool),
    c: bool,
}
"""


def test_mixed_expression(tmp_dir: Path):
    e = compare_expression(
        ["[4]Foo", "uint(0..3)", "uint(0..2)", "uint(0..4)"],
        "uint(8)",
        "a0[a1 +.. 2][a2].b.1[a3]",
        prefix=PREFIX,
        build_dir=tmp_dir,
    )
    foo = e.compile.resolve("top.Foo")

    arr = [
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
    ]

    for a1 in [0, 1]:
        for a2 in [0, 1]:
            for a3 in range(4):
                e.eval_assert([arr, a1, a2, a3], arr[a1 + a2].b[1][a3])


def test_mixed_assignment(tmp_dir: Path):
    body = """
    var v = a0;
    v[a1 +.. 2][a2].b.1[a3] = 255;
    return [v[0].b.1, v[1].b.1, v[2].b.1, v[3].b.1];
    """

    e = compare_body(
        ["[4]Foo", "uint(0..3)", "uint(0..2)", "uint(0..4)"],
        "[4][4]uint(8)",
        body,
        prefix=PREFIX,
        build_dir=tmp_dir,
    )
    foo = e.compile.resolve("top.Foo")

    arr = [
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
        foo.new(a=False, b=(False, [2, 3, 4, 5], True), c=True),
    ]

    for a1 in [0, 1]:
        for a2 in [0, 1]:
            for a3 in range(4):
                expected = [list(f.b[1]) for f in arr]
                expected[a1 + a2][a3] = 255
                e.eval_assert([arr, a1, a2, a3], expected)
