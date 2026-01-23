from pathlib import Path

from hwl_sandbox.common.compare import compare_compile


def test_enum_construction_and_match(tmp_dir: Path):
    # TODO why is the parser error so weird if we forget the val token?
    prefix = "enum ABC(T: type) { A, B, C(T), D(T) }"
    body = """
    var abc;
    match (a0) {
        0 => { abc = ABC(bool).A; }
        1 => { abc = ABC(bool).B; }
        2 => { abc = ABC(bool).C(a1); }
        3 => { abc = ABC(bool).D(a1); }
    }
    match (abc) {
        .A => { return (0, false); }
        .B => { return (1, true); }
        .C(val v) => { return (2, v); }
        .D(val v) => { return (3, !v); }
    }
    """
    e = compare_compile(["uint(0..4)", "bool"], "Tuple(uint(0..4), bool)", body, tmp_dir, prefix=prefix)
    e.eval_assert([0, False], (0, False))
    e.eval_assert([0, True], (0, False))
    e.eval_assert([1, False], (1, True))
    e.eval_assert([1, True], (1, True))
    e.eval_assert([2, False], (2, False))
    e.eval_assert([2, True], (2, True))
    e.eval_assert([3, False], (3, True))
    e.eval_assert([3, True], (3, False))
