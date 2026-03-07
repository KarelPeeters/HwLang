import hwl
import pytest
from pathlib import Path

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
    # Calling a no-payload enum variant as a function used to panic internally.
    # It should now produce a proper diagnostic error.
    # The variant is called via the generic enum (EnumNewInfer path).
    src = """
    enum Option(T: type) {
        None,
        Some(T),
    }

    module top ports() {
        const {
            val x: Option(bool) = Option.None();
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="cannot call enum variant with no payload as a constructor"):
        c = compile_custom(src)
        c.resolve("top.top")


def test_calling_no_payload_variant_as_function_via_full_type():
    # Calling a no-payload variant via the fully-specified enum type should also
    # give a proper error, not a panic. The variant is called via a fully-specified
    # enum type which gives the value directly, then calling that value as a function
    # produces a "call target must be function" error.
    src = """
    enum Option(T: type) {
        None,
        Some(T),
    }

    module top ports() {
        const {
            val x: Option(bool) = Option(bool).None();
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="call target must be function"):
        c = compile_custom(src)
        c.resolve("top.top")
