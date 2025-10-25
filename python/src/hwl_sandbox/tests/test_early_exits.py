from pathlib import Path

from hwl_sandbox.common.compare import compare_compile


def test_simple_return(tmp_dir: Path):
    src = """
    return 2;
    """
    e = compare_compile([], "int(0..8)", src, tmp_dir)
    e.eval_assert([], 2)


def test_break_compile(tmp_dir: Path):
    src = """
    var result = 0;
    for (i in 0..8) {
        result = i;
        if (i == 4) { break; }     
    }
    return result;
    """
    e = compare_compile([], "int(0..8)", src, tmp_dir)
    e.eval_assert([], 4)


def test_break_hardware(tmp_dir: Path):
    src = """
    var result = 0;
    for (i in 0..8) {
        result = i;
        if (i == a0) { break; }     
    }
    return result;
    """
    e = compare_compile(["int(-1..=8)"], "int(0..8)", src, tmp_dir)
    e.eval_assert([-1], 7)
    e.eval_assert([0], 0)
    e.eval_assert([4], 4)
    e.eval_assert([8], 7)


def test_continue_hardware(tmp_dir: Path):
    src = """
    var result = 0;
    for (i in 0..8) {
        if (i == a0) { continue; }     
        result += 1;
    }
    return result;
    """
    e = compare_compile(["int(-1..=8)"], "int(0..=8)", src, tmp_dir)
    e.eval_assert([-1], 8)
    e.eval_assert([0], 7)
    e.eval_assert([4], 7)
    e.eval_assert([8], 8)


def test_return_select(tmp_dir: Path):
    src = "if (a0) { return 1; } else { return 2; }"
    e = compare_compile(["bool"], "int(1..=2)", src, tmp_dir)
    e.eval_assert([True], 1)
    e.eval_assert([False], 2)


def test_return_loop(tmp_dir: Path):
    src = """
    for (i in 0..8) {
        if (a0 && i == 4) { return i; }     
    }
    return 0;
    """
    e = compare_compile(["bool"], "int(0..=4)", src, tmp_dir)
    e.eval_assert([False], 0)
    e.eval_assert([True], 4)


def test_mix_all(tmp_dir: Path):
    src = """
    var count = 0;
    for (i in 0..4) {
        if (i == a0) { break; }     
        if (i == a1) { continue; }
        if (i == a2) { return -1; }
        count += 1;      
    }
    return count;
    """
    e = compare_compile(
        ["int(-1..4)", "int(-1..4)", "int(-1..4)"],
        "int(-1..=4)",
        src,
        tmp_dir,
    )
    e.eval_assert([-1, -1, -1], 4)
    e.eval_assert([2, -1, -1], 2)
    e.eval_assert([-1, 2, -1], 3)
    e.eval_assert([-1, -1, 2], -1)


def test_nested_loop_break(tmp_dir: Path):
    src = """
    var count = 0;
    for (i0 in 0..4) {
        if (i0 == a0) { break; }
        for (i1 in 0..3) {
            if (i0 == a1) { break; }
            if (i1 == a2) { break; }
            count += 1;
        }
    }
    return count;
    """
    ar = "int(-1..4)"
    e = compare_compile([ar, ar, ar], "int(0..=16)", src, tmp_dir)
    e.eval_assert([-1, -1, -1], 12)
    e.eval_assert([2, -1, -1], 6)
    e.eval_assert([-1, 2, -1], 9)
    e.eval_assert([-1, -1, 2], 8)
