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

    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
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

    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
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


def test_implied_type_int_const():
    def check(ty_orig: str, cond: str, ty_true: str, ty_false: str):
        src = f"""
        import std.types.[uint, int];
        import std.util.print;
        module foo ports(p: in async uint(0..8)) {{
            comb {{
                const {{ print(type(p)); }}
                if ({cond}) {{
                    val v = p;
                    const {{ print(type(v)); }}            
                }} else {{
                    val v = p;
                    const {{ print(type(v)); }}
                }}
            }}
        }}
        """

        c = compile_custom(src)
        with c.capture_prints() as cap:
            _ = c.resolve("top.foo")
        assert cap.prints == [f"{ty_orig}\n", f"{ty_true}\n", f"{ty_false}\n"]

    check("int(0..8)", "p < 4", "int(0..4)", "int(4..8)")
    check("int(0..8)", "p <= 4", "int(0..5)", "int(5..8)")
    check("int(0..8)", "p > 4", "int(5..8)", "int(0..5)")
    check("int(0..8)", "p >= 4", "int(4..8)", "int(0..4)")

    check("int(0..8)", "4 > p", "int(0..4)", "int(4..8)")
    check("int(0..8)", "4 >= p", "int(0..5)", "int(5..8)")
    check("int(0..8)", "4 < p", "int(5..8)", "int(0..5)")
    check("int(0..8)", "4 <= p", "int(4..8)", "int(0..4)")
