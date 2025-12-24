from pathlib import Path

import pytest

from hwl_sandbox.common.compare import compare_expression, compare_compile


def test_array_literal_empty_const(tmp_dir: Path):
    e = compare_expression([], "[0]bool", "[]", tmp_dir)
    e.eval_assert([], [])


def test_array_pass_empty_hw(tmp_dir: Path):
    e = compare_expression(["[0]bool"], "[0]bool", "a0", tmp_dir)
    e.eval_assert([[]], [])


def test_array_literal_bool_const(tmp_dir: Path):
    v = [False, False, True]
    e = compare_expression([], "[3]bool", str(v).lower(), tmp_dir)
    e.eval_assert([], v)


def test_array_literal_bool_const_2d(tmp_dir: Path):
    v = [[False, False, True], [True, False, False]]
    e = compare_expression([], "[2][3]bool", str(v).lower(), tmp_dir)
    e.eval_assert([], v)


def test_array_literal_tuple(tmp_dir: Path):
    # Test array literals of compound types
    e = compare_expression([], "[2](bool, uint(4))", "[(false, 0), (true, 1)]", tmp_dir)
    e.eval_assert([], [(False, 0), (True, 1)])


def test_array_index_bool_const(tmp_dir: Path):
    v = [False, False, True]
    e = compare_expression(["uint(0..3)"], "bool", f"({str(v).lower()})[a0]", tmp_dir)
    for i in range(len(v)):
        e.eval_assert([i], v[i])


def test_array_index_bool_hw(tmp_dir: Path):
    e = compare_expression(["[3]bool", "uint(0..3)"], "bool", f"a0[a1]", tmp_dir)

    vs = [[False, False, True], [True, True, False]]
    for v in vs:
        for i in range(len(v)):
            e.eval_assert([v, i], v[i])


def test_array_index_bool_hw_2d(tmp_dir: Path):
    e = compare_expression(["[2][3]bool", "uint(0..2)", "uint(0..3)"], "bool", f"a0[a1][a2]", tmp_dir)

    vs = [
        [[False] * 3] * 2,
        [[True] * 3] * 2,
        [[False, False, True], [True, True, False]]
    ]

    for v in vs:
        for i in range(len(v)):
            for j in range(len(v[i])):
                e.eval_assert([v, i, j], v[i][j])


def test_array_index_int_const(tmp_dir: Path):
    # TODO more numbers, to test all bits
    v = [4, 5, 6]
    e = compare_expression(["uint(0..3)"], "int(8)", f"({str(v).lower()})[a0]", tmp_dir)
    for i in range(len(v)):
        e.eval_assert([i], v[i])


def test_array_index_int_hw(tmp_dir: Path):
    e = compare_expression(["[3]int(8)", "uint(0..3)"], "int(8)", f"a0[a1]", tmp_dir)

    vs = [[4, 5, 6], [7, 8, 9]]
    for v in vs:
        for i in range(len(v)):
            e.eval_assert([v, i], v[i])


def test_array_index_int_hw_single(tmp_dir: Path):
    e = compare_expression(["[1]int(8)", "uint(0..1)"], "int(8)", f"a0[a1]", tmp_dir)

    vs = [[4], [79]]
    for v in vs:
        e.eval_assert([v, 0], v[0])


def test_array_slice_empty_empty_start_const(tmp_dir: Path):
    e = compare_expression([f"[0]int(8)"], "[0]int(8)", "a0[0+..0]", tmp_dir)
    e.eval_assert([[]], [])


def test_array_slice_empty_empty_start_hw(tmp_dir: Path):
    e = compare_expression([f"[0]int(8)", "uint(0)"], "[0]int(8)", "a0[a1+..0]", tmp_dir)
    e.eval_assert([[], 0], [])


def test_array_slice_real_empty_start_hw(tmp_dir: Path):
    e = compare_expression([f"[3]int(8)", "uint(0..3)"], "[0]int(8)", "a0[a1+..0]", tmp_dir)
    for i in range(3):
        e.eval_assert([[1, 2, 3], i], [])


def test_array_slice_int_const_range(tmp_dir: Path):
    e = compare_expression(["[4]int(8)"], "[2]int(8)", "a0[1..3]", tmp_dir)
    e.eval_assert([[4, 5, 6, 7]], [5, 6])


@pytest.mark.parametrize("slice_len", range(4 + 1))
def test_array_slice_int_hw(tmp_dir: Path, slice_len: int):
    array_len = 4
    vs = [[4, 5, 6, 7], [8, 9, 10, 11]]

    tmp_dir_iter = tmp_dir / f"{slice_len}"
    start_end = array_len - slice_len + 1
    start_ty = f"uint(0..{start_end})"

    print(f"Testing slice length {slice_len}, start will have type {start_ty}")

    e = compare_expression(
        [f"[{array_len}]int(8)", start_ty],
        f"[{slice_len}]int(8)",
        f"a0[a1 +.. {slice_len}]",
        tmp_dir_iter
    )

    for v in vs:
        for i in range(start_end):
            e.eval_assert([v, i], v[i:i + slice_len])


def test_array_assign_bool(tmp_dir: Path):
    body = """
    var v: [3]bool = a0;
    v[a1] = a2;
    return v;
    """
    e = compare_compile(["[3]bool", "uint(0..3)", "bool"], "[3]bool", body, tmp_dir)
    e.eval_assert([[False, False, False], 0, True], [True, False, False])
    e.eval_assert([[False, False, False], 1, True], [False, True, False])
    e.eval_assert([[False, False, False], 2, True], [False, False, True])


def test_array_assign_int(tmp_dir: Path):
    body = """
    var v: [3]uint(8) = a0;
    v[a1] = a2;
    return v;
    """
    e = compare_compile(["[3]uint(8)", "uint(0..3)", "uint(8)"], "[3]uint(8)", body, tmp_dir)
    e.eval_assert([[2, 3, 4], 0, 5], [5, 3, 4])
    e.eval_assert([[2, 3, 4], 1, 5], [2, 5, 4])
    e.eval_assert([[2, 3, 4], 2, 5], [2, 3, 5])


def test_array_assign_slice(tmp_dir: Path):
    body = """
    var v: [4]uint(8) = a0;
    v[a1 +.. 2] = a2;
    return v;
    """
    e = compare_compile(["[4]uint(8)", "uint(0..3)", "[2]uint(8)"], "[4]uint(8)", body, tmp_dir)

    v = [2, 3, 4, 5]
    s = [6, 7]
    for i in range(2 + 1):
        c = list(v)
        c[i:i + 2] = s
        e.eval_assert([v, i, s], c)


def test_array_assign_slice_twice(tmp_dir: Path):
    body = """
    var v: [4]uint(8) = a0;
    v[1..][a1 +.. 2] = a2;
    return v;
    """
    e = compare_compile(["[4]uint(8)", "uint(0..2)", "[2]uint(8)"], "[4]uint(8)", body, tmp_dir)

    v = [2, 3, 4, 5]
    s = [6, 7]
    for i in range(1 + 1):
        c = list(v)
        c[1 + i:1 + i + 2] = s
        e.eval_assert([v, i, s], c)

# TODO test array assignments
# TODO test array slice copy from one array to another
# TODO test array concat, including spread operator
# TODO test array repeat
# TODO test signal array slicing and assigning
# TODO test and fix tuples
