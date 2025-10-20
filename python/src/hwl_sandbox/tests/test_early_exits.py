from pathlib import Path

from hwl_sandbox.common.compare import compare_compile


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
    e = compare_compile(["int(0..=8)"], "int(0..8)", src, tmp_dir)
    e.eval_assert([0], 0)
    e.eval_assert([4], 4)
    e.eval_assert([8], 7)


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
