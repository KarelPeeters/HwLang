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
    ty = "0..8"

    check_unary_cond(ty, "p < 4", "0..4", "4..8")
    check_unary_cond(ty, "4 > p", "0..4", "4..8")

    check_unary_cond(ty, "p <= 4", "0..5", "5..8")
    check_unary_cond(ty, "4 >= p", "0..5", "5..8")

    check_unary_cond(ty, "p > 4", "5..8", "0..5")
    check_unary_cond(ty, "4 < p", "5..8", "0..5")

    check_unary_cond(ty, "p >= 4", "4..8", "0..4")
    check_unary_cond(ty, "4 <= p", "4..8", "0..4")

    check_unary_cond(ty, "p == 4", "4..5", "0..4, 5..8")
    check_unary_cond(ty, "4 == p", "4..5", "0..4, 5..8")

    check_unary_cond(ty, "p != 4", "0..4, 5..8", "4..5")
    check_unary_cond(ty, "4 != p", "0..4, 5..8", "4..5")


def test_implied_type_int_int():
    ty_a = "0..8"
    ty_b = "2..6"

    check_binary_cond(ty_a, ty_b, "a < b", "0..5", "2..8")
    check_binary_cond(ty_a, ty_b, "b > a", "0..5", "2..8")

    check_binary_cond(ty_a, ty_b, "a <= b", "0..6", "3..8")
    check_binary_cond(ty_a, ty_b, "b >= a", "0..6", "3..8")

    check_binary_cond(ty_a, ty_b, "a > b", "3..8", "0..6")
    check_binary_cond(ty_a, ty_b, "b < a", "3..8", "0..6")

    check_binary_cond(ty_a, ty_b, "a >= b", "2..8", "0..5")
    check_binary_cond(ty_a, ty_b, "b <= a", "2..8", "0..5")


def test_implied_type_operators():
    # TODO make these stricter once we have better range analysis for composite conditions
    check_unary_cond("0..8", "2 <= p && p < 6", "2..6", "0..8")
    check_unary_cond("0..8", "p < 2 || 6 <= p", "0..8", "2..6")
    check_unary_cond("0..8", "2 <= p ^^ p < 6", "0..8", "0..8")
    check_unary_cond("0..8", "!(2 <= p)", "0..2", "2..8")


def check_unary_cond(ty: str, cond: str, ty_true: str, ty_false: str):
    src = f"""
    import std.types.[uint, int];
    import std.util.print;
    module foo ports(p: in async int({ty})) {{
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
    assert cap.prints == [f"int({ty})\n", f"int({ty_true})\n", f"int({ty_false})\n"]


def check_binary_cond(ty_a: str, ty_b: str, cond: str, ty_true_a: str, ty_false_a: str):
    src = f"""
    import std.types.[uint, int];
    import std.util.print;
    module foo ports(a: in async int({ty_a}), b: in async int({ty_b})) {{
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
        f"int({ty_a})\n", f"int({ty_b})\n",
        f"int({ty_true_a})\n", f"int({ty_b})\n",
        f"int({ty_false_a})\n", f"int({ty_b})\n"
    ]


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


def test_imply_non_zero_div():
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


def test_imply_nested():
    src = """
    import std.types.uint;
    module foo ports(p: in async uint(0..8)) {
        comb {
            if (p < 6) {
                if (p >= 2) {
                    var v: uint(2..6) = p;   
                }
            }
        }
    }
    """
    c = compile_custom(src)
    _ = c.resolve("top.foo")
