from pathlib import Path

from hwl_sandbox.common.compare import compare_compile


def test_for_int(tmp_dir: Path):
    src = """
    var s = 0;
    for (i in 0..8) {
        s += a0;
    }
    return s;
    """
    c = compare_compile(["uint(0..=3)"], "uint(0..=3*8)", src, tmp_dir)

    for i in range(4):
        c.eval_assert([i], i * 8)


def test_for_compile_array(tmp_dir: Path):
    src = """
    var s = 0;
    for (i in [0, 1, 2, 3]) {
        s += i * a0;
    }
    return s;
    """
    c = compare_compile(["uint(0..=3)"], "uint(0..=3*6)", src, tmp_dir)

    for i in range(4):
        c.eval_assert([i], i * 6)


def test_for_hardware_array(tmp_dir: Path):
    src = """
    var s = 0;
    for (i in a0) {
        s += i;
    }
    return s;
    """
    c = compare_compile(["[4]uint(0..=3)"], "uint(0..=3*4)", src, tmp_dir)

    c.eval_assert([[0, 0, 0, 0]], 0)
    c.eval_assert([[0, 1, 2, 3]], 6)
    c.eval_assert([[3, 3, 3, 3]], 4 * 3)
