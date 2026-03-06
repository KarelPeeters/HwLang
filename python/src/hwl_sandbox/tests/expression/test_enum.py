from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_body
from hwl_sandbox.common.util import compile_custom


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
    e = compare_body(["uint(0..4)", "bool"], "Tuple(uint(0..4), bool)", body, tmp_dir, prefix=prefix)
    e.eval_assert([0, False], (0, False))
    e.eval_assert([0, True], (0, False))
    e.eval_assert([1, False], (1, True))
    e.eval_assert([1, True], (1, True))
    e.eval_assert([2, False], (2, False))
    e.eval_assert([2, True], (2, True))
    e.eval_assert([3, False], (3, True))
    e.eval_assert([3, True], (3, False))


def test_calling_no_payload_variant_as_function():
    """Calling a no-payload enum variant as a function should give a clear error, not panic."""
    # Without args
    src = """
    enum Option(T: type) { None, Some(T) }
    fn test() -> Option(bool) {
        val x: Option(bool) = Option.None();
        return x;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="takes no arguments and cannot be called"):
        compile_custom(src).resolve("top.test")()

    # With args
    src2 = """
    enum Option(T: type) { None, Some(T) }
    fn test() -> Option(bool) {
        val x: Option(bool) = Option.None(true);
        return x;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="takes no arguments and cannot be called"):
        compile_custom(src2).resolve("top.test")()
