import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_signal_does_not_fit():
    source = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            var v: uint(0..4) = p;
        }
    }
    """
    c = compile_custom(source)

    with pytest.raises(hwl.DiagnosticException, match="value does not fit in type"):
        _ = c.resolve("top.foo")


def test_signal_checked():
    source = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            if (p < 4) {
                var v: uint(0..4) = p;
            }
        }
    }
    """
    c = compile_custom(source)
    _ = c.resolve("top.foo")


def test_variable_does_not_fit():
    source = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            var s = p;
            var v: uint(0..4) = s;
        }
    }
    """
    c = compile_custom(source)

    with pytest.raises(hwl.DiagnosticException, match="value does not fit in type"):
        _ = c.resolve("top.foo")


def test_variable_checked():
    source = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            var s = p;
            if (s < 4) {
                var v: uint(0..4) = s;
            }
        }
    }
    """
    c = compile_custom(source)
    _ = c.resolve("top.foo")
