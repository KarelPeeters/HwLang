from pathlib import Path

from khdl_sandbox.common.compare import compare_body


def test_for_int(tmp_dir: Path):
    src = """
    var s = 0;
    for (i in 0..8) {
        s += a0;
    }
    return s;
    """
    c = compare_body(["uint(0..=3)"], "uint(0..=3*8)", src, tmp_dir)

    for i in range(4):
        c.eval_assert([i], i * 8)


def test_for_tuple_compile(tmp_dir: Path):
    src = """
    var i = 0;
    var r = a0;
    for (v in (2, true, 5, uint(4))) {
        if (i == 0 || i == 2) { r += v; }
        if (i == 1) { if (v) { r += 1; } }
        if (i == 3) { var x: v = 7; r += x; }
        i += 1;
    }
    return r;
    """
    c = compare_body(["uint(4)"], "uint(8)", src, tmp_dir)
    c.eval_assert([0], 15)
    c.eval_assert([3], 18)
    c.eval_assert([15], 30)


def test_for_tuple_mixed(tmp_dir: Path):
    src = """
    var i = 0;
    var r = 0;
    for (v in (a0, true, 7, a1, bool)) {
        if (i == 0) { if (v) { r += 1; } }
        if (i == 1) { if (v) { r += 1; } }
        if (i == 2 || i == 3) { r += v; }
        if (i == 4) { var _: v = false; }
        i += 1;
        r += 1;
    }
    return r;
    """
    c = compare_body(["bool", "uint(4)"], "uint(8)", src, tmp_dir)
    c.eval_assert([False, 2], 15)
    c.eval_assert([False, 3], 16)
    c.eval_assert([True, 2], 16)
    c.eval_assert([True, 15], 29)


def test_for_tuple_hardware(tmp_dir: Path):
    src = """
    var i = 0;
    var r = 0;
    for (v in a0) {
        if (i == 0) { if (v) { r += 1; } }
        if (i == 1) { r += v; }
        i += 1;
    }
    return r;
    """
    c = compare_body(["Tuple(bool, uint(4))"], "uint(8)", src, tmp_dir)
    c.eval_assert([(False, 2)], 2)
    c.eval_assert([(False, 3)], 3)
    c.eval_assert([(True, 2)], 3)
    c.eval_assert([(True, 15)], 16)


def test_for_compile_array(tmp_dir: Path):
    src = """
    var s = 0;
    for (i in [0, 1, 2, 3]) {
        s += i * a0;
    }
    return s;
    """
    c = compare_body(["uint(0..=3)"], "uint(0..=3*6)", src, tmp_dir)

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
    c = compare_body(["[4]uint(0..=3)"], "uint(0..=3*4)", src, tmp_dir)

    c.eval_assert([[0, 0, 0, 0]], 0)
    c.eval_assert([[0, 1, 2, 3]], 6)
    c.eval_assert([[3, 3, 3, 3]], 4 * 3)
