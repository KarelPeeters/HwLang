from pathlib import Path

from hwl_sandbox.common.compare import compare_body
from hwl_sandbox.common.util import compile_custom


# TODO also test these at signal level, with multiple different processes driving different splits
def test_struct_assign_field_const(tmp_dir: Path):
    src = """
    struct Pair { x: uint(8), y: bool }
    fn f() -> Pair {
        var v = Pair.new(x=0, y=false);
        v.x = 8;
        return v;
    }
    """

    c = compile_custom(src)
    pair = c.resolve("top.Pair")
    f = c.resolve("top.f")

    assert f() == pair.new(x=8, y=False)


def test_struct_assign_field_cond(tmp_dir: Path):
    prefix = """
    struct Pair { x: uint(8), y: bool }
    """
    src = """
    var v = Pair.new(x=0, y=false);
    if (a0) {
        v.x = a1;
    }
    if (a2) {
        v.y = a3;
    }
    return (v.x, v.y);
    """
    e = compare_body(["bool", "uint(8)", "bool", "bool"], "Tuple(uint(8), bool)", src, tmp_dir, prefix=prefix)

    bools = [False, True]
    for a0 in bools:
        for a1 in range(2):
            for a2 in bools:
                for a3 in bools:
                    res_x = False
                    res_y = False
                    if a0:
                        res_x = a1
                    if a2:
                        res_y = a3

                    e.eval_assert([a0, a1, a2, a3], (res_x, res_y))


def test_tuple_assign_index_const(tmp_dir: Path):
    src = """
    fn f() -> Tuple(uint(8), bool) {
        var v = (0, false);
        v.0 = 8;
        return v;
    }
    """

    c = compile_custom(src)
    f = c.resolve("top.f")

    assert f() == (8, False)


def test_tuple_assign_index_cond(tmp_dir: Path):
    src = """
    var v = (0, false);
    if (a0) {
        v.0 = a1;
    }
    if (a2) {
        v.1 = a3;
    }
    return v;
    """
    e = compare_body(["bool", "uint(8)", "bool", "bool"], "Tuple(uint(8), bool)", src, tmp_dir)

    bools = [False, True]
    for a0 in bools:
        for a1 in range(2):
            for a2 in bools:
                for a3 in bools:
                    res_x = False
                    res_y = False
                    if a0:
                        res_x = a1
                    if a2:
                        res_y = a3

                    e.eval_assert([a0, a1, a2, a3], (res_x, res_y))
