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


def test_enum_infer_different():
    src = """
    enum Foo(T: type) {
        None,
        Some(T),
    }
    enum Bar(T: type) {
        None,
        Some(bool),
    }
    fn f() {
        val v: Foo(bool) = Bar.Some(true);
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="enum expected type mismatch"):
        f()


def test_enum_cannot_infer():
    src = """
    enum Foo(T: type) {
        None,
        Some(T),
    }
    fn f() {
        val v = Foo.None;
    }
    fn g() {
        val v = Foo.Some(false);
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    g = c.resolve("top.g")
    with pytest.raises(hwl.DiagnosticException, match="cannot infer enum parameters"):
        f()
    with pytest.raises(hwl.DiagnosticException, match="cannot infer enum parameters"):
        g()


def test_enum_infer_no_payload():
    src = """
    enum Foo(T: type) {
        None,
        Some(T),
    }
    fn f() {
        val v: Foo(bool) = Foo.None(false);
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="cannot infer enum parameters"):
        f()


def test_enum_infer_non_existing():
    src = """
    enum Foo(T: type) {
        None,
        Some(T),
    }
    fn f() {
        val v = Foo.NonExisting;
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="enum variant `NonExisting` not found"):
        f()


def test_enum_infer_maybe_existing():
    src = """
    enum Foo(T: type, c: bool) {
        None,
        Some(T),
        if (c) {
            MaybeExisting(T),
        }
    }
    fn f() {
        val v = Foo.MaybeExisting;
    }
    """
    f = compile_custom(src).resolve("top.f")
    f()


def test_enum_mixed_payload():
    src = """
    enum Foo(c: bool) {
        if (c) {
            Var,
        } else {
            Var(bool),
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="payload must be consistent"):
        _ = compile_custom(src).resolve("top.Foo")


def test_enum_empty():
    src = """  
    enum Empty {}
    enum NonEmpty { A, B, C }
    module top_empty ports(x: in async Empty) {}    
    module top_non_empty ports(x: in async NonEmpty) {}    
    """
    with pytest.raises(hwl.DiagnosticException, match="port type must be representable in hardware"):
        _ = compile_custom(src).resolve("top.top_empty")
    _ = compile_custom(src).resolve("top.top_non_empty")

# TODO test that we can actually get enum instances back from
#   * compile-time functions
#   * module port outputs/inputs
