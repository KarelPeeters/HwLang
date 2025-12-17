import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_signal_does_not_fit():
    src = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            var v: uint(0..4) = p;
        }
    }
    """
    c = compile_custom(src)

    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        _ = c.resolve("top.foo")


def test_signal_checked():
    src = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            if (p < 4) {
                var v: uint(0..4) = p;
            }
        }
    }
    """
    c = compile_custom(src)
    _ = c.resolve("top.foo")


def test_variable_does_not_fit():
    src = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            var s = p;
            var v: uint(0..4) = s;
        }
    }
    """
    c = compile_custom(src)

    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        _ = c.resolve("top.foo")


def test_variable_checked():
    src = """
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
    c = compile_custom(src)
    _ = c.resolve("top.foo")


def test_implied_type_int_const():
    def check(cond: str, ty_true: str, ty_false: str):
        ty_orig = "int(0..8)"
        src = f"""
        import std.types.[uint, int];
        import std.util.print;
        module foo ports(p: in async {ty_orig}) {{
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
        assert cap.prints == [f"{ty_orig}\n", f"int({ty_true})\n", f"int({ty_false})\n"]

    check("p < 4", "0..4", "4..8")
    check("p <= 4", "0..5", "5..8")
    check("p > 4", "5..8", "0..5")
    check("p >= 4", "4..8", "0..4")

    check("4 > p", "0..4", "4..8")
    check("4 >= p", "0..5", "5..8")
    check("4 < p", "5..8", "0..5")
    check("4 <= p", "4..8", "0..4")


def test_implied_type_int_int():
    def check(cond: str, ty_true_a: str, ty_false_a: str):
        ty_orig_a = "int(0..8)"
        ty_orig_b = "int(2..6)"
        src = f"""
        import std.types.[uint, int];
        import std.util.print;
        module foo ports(a: in async {ty_orig_a}, b: in async {ty_orig_b}) {{
            comb {{
                const {{ print(type(a)); print(type(b)); }}
                if ({cond}) {{
                    val va = a;
                    val vb = b;
                    const {{ print(type(va)); print(type(vb)); }}
                }} else {{
                    val va = a;
                    val vb = b;
                    const {{ print(type(va)); print(type(vb)); }}
                }}
            }}
        }}
        """

        c = compile_custom(src)
        with c.capture_prints() as cap:
            _ = c.resolve("top.foo")
        assert cap.prints == [
            f"{ty_orig_a}\n", f"{ty_orig_b}\n",
            f"int({ty_true_a})\n", f"{ty_orig_b}\n",
            f"int({ty_false_a})\n", f"{ty_orig_b}\n"
        ]

    # ty_orig_a = "int(0..8)"
    # ty_orig_b = "int(2..6)"
    check("a < b", "0..5", "2..8")
    check("a <= b", "0..6", "3..8")
    check("a > b", "3..8", "0..6")
    check("a >= b", "2..8", "0..5")

    check("b > a", "0..5", "2..8")
    check("b >= a", "0..6", "3..8")
    check("b < a", "3..8", "0..6")
    check("b <= a", "2..8", "0..5")


def test_bool_implies_itself():
    src = """
    import std.types.bool;
    import std.util.print;
    module foo ports(p: in async bool) {
        comb {
            if (p) {
                const { print("true"); }
                if (p) {
                    const { print("true.true"); }
                } else {
                    const { print("true.false"); }
                }
            } else {
                const { print("false"); }
                if (p) {
                    const { print("false.true"); }
                } else {
                    const { print("false.false"); }
                }
            }
        }
    }
    """
    c = compile_custom(src)
    with c.capture_prints() as cap:
        _ = c.resolve("top.foo")
    assert cap.prints == [
        "true\n",
        "true.true\n",
        "false\n",
        "false.false\n",
    ]


def test_imply_non_zero():
    src_raw = """
    import std.types.int;
    module foo ports(x: in async int(-4..=4), y: out async int(-4..=4)) {
        comb {
            y = 4 / x;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="division by zero"):
        _ = compile_custom(src_raw).resolve("top.foo")

    src_checked = """
    import std.types.int;
    module foo ports(x: in async int(-4..=4), y: out async int(-4..=4)) {
        comb {
            if (x != 0) {
                y = 4 / x;
            } else {
                y = 0;
            }
        }
    }
    """
    _ = compile_custom(src_checked).resolve("top.foo")
