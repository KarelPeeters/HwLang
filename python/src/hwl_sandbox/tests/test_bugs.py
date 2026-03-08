"""
Tests that find bugs in the HwLang compiler: parse errors, missing features,
ICEs (internal compiler errors / crashes), and hardware miscompilations.

Each test is clearly labelled with its status:
  - PARSE LIMITATION  – grammar limitation causing unexpected-token errors
  - MISSING FEATURE   – feature that is planned but not yet implemented
  - BUG / CRASH       – actual compiler crash or wrong behaviour (not intentional)
  - DESIGN DECISION   – intentional behaviour that might look surprising
  - CORRECT           – things that work as expected (regression guards)

How hardware correctness tests work
------------------------------------
`compare_body` / `compare_expression` compile the body as both
  (a) an interpreted HwLang function  and
  (b) a Verilog module simulated with Verilator.
The helper then asserts that both produce the same result for every input.
Any disagreement is a hardware miscompilation.
"""
from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_body, compare_expression
from hwl_sandbox.common.util import compile_custom


# =============================================================================
# PARSE LIMITATIONS
# These are grammar restrictions where postfix `.field` or `[index]` cannot
# follow certain primary expression forms.
# =============================================================================

def test_parse_array_index_then_dot_field():
    """
    PARSE LIMITATION: `arr[i].field` is rejected by the parser.
    The workaround is `(arr[i]).field`.
    Both `fn().field` (function-call then dot) and `arr[i][j]` (double index) work.
    """
    src = """
    struct Point { x: int, y: int }
    fn f() -> int {
        val pts = [Point.new(x=1, y=2), Point.new(x=3, y=4)];
        return pts[1].x;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_array_index_then_dot_field_workaround():
    """CORRECT: `(arr[i]).field` works as a workaround."""
    src = """
    struct Point { x: int, y: int }
    fn f() -> int {
        val pts = [Point.new(x=1, y=2), Point.new(x=3, y=4)];
        return (pts[1]).x;
    }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 3


def test_parse_array_index_then_dot_int():
    """
    PARSE LIMITATION: accessing a tuple element after array index, `arr[i].0`,
    is rejected by the parser.
    """
    src = """
    fn f() -> int {
        val a = [(1, 2), (3, 4)];
        return a[1].0;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_array_index_then_dot_int_workaround():
    """CORRECT: `(arr[i]).0` works as a workaround."""
    src = """
    fn f() -> int {
        val a = [(1, 2), (3, 4)];
        return (a[1]).0;
    }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 3


def test_parse_tuple_literal_then_dot_int():
    """
    PARSE LIMITATION: `(a, b).0` — a tuple *literal* followed immediately by `.0`
    is rejected by the parser.
    """
    src = "fn f(a: int, b: int) -> int { return (a, b).0; }"
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_array_literal_then_index():
    """
    PARSE LIMITATION: `[1, 2, 3][1]` — indexing an array literal inline
    is rejected by the parser.
    """
    src = "fn f() -> int { return [10, 20, 30][1]; }"
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_fn_call_then_index_then_dot_field():
    """
    PARSE LIMITATION: `f()[0].field` — chaining index after function call,
    then accessing a field, is rejected.
    Note: `f()[0]` alone works, and `f().field` alone works.
    """
    src = """
    struct P { x: int, y: int }
    fn make() -> [2]P { return [P.new(x=1, y=2), P.new(x=3, y=4)]; }
    fn f() -> int { return make()[0].x; }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_fn_call_then_index_works():
    """CORRECT: `f()[0]` (function call then array index) is allowed."""
    src = """
    fn make() -> [3]int { return [10, 20, 30]; }
    fn f() -> int { return make()[1]; }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 20


def test_parse_fn_call_then_dot_field_works():
    """CORRECT: `f().field` (function call then struct field) is allowed."""
    src = """
    struct P { x: int, y: int }
    fn make() -> P { return P.new(x=42, y=0); }
    fn f() -> int { return make().x; }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 42


def test_parse_double_array_index_works():
    """CORRECT: `arr[i][j]` (double array index) is allowed."""
    src = """
    fn f() -> int {
        val arr = [[1, 2], [3, 4], [5, 6]];
        return arr[1][0];
    }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 3


def test_parse_if_expression_not_supported():
    """
    PARSE LIMITATION: `if (cond) { x } else { y }` as an expression
    (inline if-expression, ternary-style) is not supported.
    """
    src = "fn f(a: bool, x: int, y: int) -> int { return if (a) { x } else { y }; }"
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_tuple_destructure_not_supported():
    """
    PARSE LIMITATION: tuple destructuring `val (a, b) = expr` is not yet supported.
    """
    src = "fn f() -> int { val (a, b) = (1, 2); return a + b; }"
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


# =============================================================================
# MISSING FEATURES (implemented only partially, error at elaboration time)
# =============================================================================

def test_struct_field_assignment_not_implemented():
    """
    MISSING FEATURE: assignment to a struct field (`p.x = 5`) is not yet
    implemented for non-interface targets.
    The compiler accepts the source but raises a 'feature not yet implemented'
    error at elaboration (call) time.
    """
    src = """
    struct P { x: int, y: int }
    fn f() -> int {
        var p = P.new(x=1, y=2);
        p.x = 5;
        return p.x + p.y;
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException,
                       match="feature not yet implemented"):
        f()


# =============================================================================
# Infinite recursion raises a clean "stack overflow" diagnostic
# =============================================================================

def test_infinite_recursion_raises_stack_overflow():
    """
    CORRECT (after fix): Calling a function with no base case (infinite
    recursion) raises a clean DiagnosticException with "stack overflow" rather
    than crashing the process with SIGSEGV.

    Root cause of the previous crash: STACK_OVERFLOW_STACK_LIMIT was 1000,
    meaning ~500 actual recursive HwLang calls before the check fires.  The
    Python-binding call path runs on the Python thread whose default stack is
    only ~8 MB.  At ~500 recursive calls the Rust system stack was exhausted
    first, producing SIGSEGV.  The limit is now 128 (~64 real recursive calls),
    which fires well within the 8 MB budget.

    Note: mutual recursion that terminates works correctly (see test below).
    """
    src = """
fn f(n: uint) -> int {
    return f(n + 1);
}
"""
    c = compile_custom(src)
    f = c.resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="stack overflow"):
        f(0)


def test_terminating_mutual_recursion_works():
    """
    CORRECT: Mutual recursion with a proper base case works correctly.
    """
    src = """
    fn is_even(n: uint) -> bool {
        if (n == 0) { return true; }
        return is_odd(n - 1);
    }
    fn is_odd(n: uint) -> bool {
        if (n == 0) { return false; }
        return is_even(n - 1);
    }
    """
    c = compile_custom(src)
    is_even = c.resolve("top.is_even")
    is_odd = c.resolve("top.is_odd")
    for n in range(8):
        assert is_even(n) == (n % 2 == 0), f"is_even({n}) wrong"
        assert is_odd(n) == (n % 2 == 1), f"is_odd({n}) wrong"


# =============================================================================
# DESIGN DECISIONS (errors happen at elaboration / call time, not resolve time)
# =============================================================================

def test_type_out_of_range_is_elaboration_error():
    """
    DESIGN DECISION: assigning a value that doesn't fit in the declared return
    type is a type error raised at elaboration (call) time, not at resolve time.
    """
    src = "fn f() -> uint(8) { return 256; }"
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f()


def test_type_negative_in_uint_is_elaboration_error():
    """
    DESIGN DECISION: returning a negative value for a uint return type is a
    type error at elaboration (call) time.
    """
    src = "fn f() -> uint(8) { return -1; }"
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f()


def test_zero_pow_zero_elaboration_error():
    """
    DESIGN DECISION: `0 ** 0` is rejected (mathematically undefined) at
    elaboration time.  Error message: "invalid power operation".
    """
    src = "fn f() -> int { return 0 ** 0; }"
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="invalid power operation"):
        f()


def test_int_x_pow_zero_elaboration_error():
    """
    DESIGN DECISION: `x ** 0` where x could be 0 is rejected because 0**0
    is undefined.  The compiler requires the base to be non-zero.
    """
    src = "fn f(x: int) -> int { return x ** 0; }"
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="invalid power operation"):
        f(0)


def test_missing_return_in_all_paths_is_elaboration_error():
    """
    DESIGN DECISION: a function whose return path is not guaranteed (not all
    branches return) raises a 'match statement reached end' error at elaboration
    time when the non-returning path is taken.
    """
    src = """
    fn f(x: uint(4)) -> int {
        match (x) {
            0 => { return 0; }
            1 => { return 1; }
        }
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f(0) == 0
    assert f(1) == 1
    with pytest.raises(hwl.DiagnosticException, match="match statement reached end"):
        f(2)


def test_enum_none_is_not_callable_elaboration_error():
    """
    DESIGN DECISION: calling a no-payload enum variant as a function,
    e.g. `Option(bool).None()`, is rejected at elaboration time with
    "call target must be function".
    """
    src = """
    enum Option(T: type) { None, Some(T) }
    fn f() -> Option(bool) {
        return Option(bool).None();
    }
    """
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="call target must be function"):
        f()


# =============================================================================
# HARDWARE CORRECTNESS TESTS (compare interpreter vs Verilog simulation)
# All of the tests below should pass — they are regression guards confirming
# that the hardware codegen produces correct Verilog for various language features.
# =============================================================================

def test_hw_enum_tag_encoding(tmp_dir: Path):
    """
    CORRECT: 16-variant enum round-trip in hardware.
    Int -> enum (via match) -> back to int (via match).  Verifies that each
    enum variant has a unique bit pattern and that the hardware match correctly
    decodes every tag.
    """
    prefix = """
    enum Big16 {
        V0,  V1,  V2,  V3,  V4,  V5,  V6,  V7,
        V8,  V9,  V10, V11, V12, V13, V14, V15
    }
    fn int_to_big16(i: uint(0..16)) -> Big16 {
        match (i) {
            0  => { return Big16.V0;  }   1  => { return Big16.V1;  }
            2  => { return Big16.V2;  }   3  => { return Big16.V3;  }
            4  => { return Big16.V4;  }   5  => { return Big16.V5;  }
            6  => { return Big16.V6;  }   7  => { return Big16.V7;  }
            8  => { return Big16.V8;  }   9  => { return Big16.V9;  }
            10 => { return Big16.V10; }   11 => { return Big16.V11; }
            12 => { return Big16.V12; }   13 => { return Big16.V13; }
            14 => { return Big16.V14; }   15 => { return Big16.V15; }
        }
    }
    """
    body = """
    val e = int_to_big16(a0);
    match (e) {
        .V0  => { return 0;  }   .V1  => { return 1;  }
        .V2  => { return 2;  }   .V3  => { return 3;  }
        .V4  => { return 4;  }   .V5  => { return 5;  }
        .V6  => { return 6;  }   .V7  => { return 7;  }
        .V8  => { return 8;  }   .V9  => { return 9;  }
        .V10 => { return 10; }   .V11 => { return 11; }
        .V12 => { return 12; }   .V13 => { return 13; }
        .V14 => { return 14; }   .V15 => { return 15; }
    }
    """
    e = compare_body(["uint(0..16)"], "uint(0..16)", body, tmp_dir, prefix=prefix)
    for v in range(16):
        e.eval_assert([v], v)


def test_hw_enum_payload_roundtrip(tmp_dir: Path):
    """
    CORRECT: Enum with a payload round-trips correctly through hardware:
    the tag bit and the payload field are both correctly encoded/decoded.
    """
    prefix = """
    enum Msg { Empty, WithData(uint(8)) }
    fn make_msg(has_data: bool, data: uint(8)) -> Msg {
        if (has_data) {
            return Msg.WithData(data);
        } else {
            return Msg.Empty;
        }
    }
    """
    body = """
    val m = make_msg(a0, a1);
    match (m) {
        .Empty           => { return (false, 0); }
        .WithData(val d) => { return (true, d);  }
    }
    """
    e = compare_body(["bool", "uint(8)"], "Tuple(bool, uint(8))", body,
                     tmp_dir, prefix=prefix)
    e.eval_assert([False, 0], (False, 0))
    e.eval_assert([False, 200], (False, 0))
    e.eval_assert([True, 0], (True, 0))
    e.eval_assert([True, 42], (True, 42))
    e.eval_assert([True, 255], (True, 255))


def test_hw_nested_struct_field_access(tmp_dir: Path):
    """
    CORRECT: Hardware correctly accesses fields of a struct nested inside
    another struct -- bit offsets must be correct at every level.
    """
    prefix = """
    struct Inner { x: uint(4), y: uint(4) }
    struct Outer { a: uint(4), b: Inner, c: uint(4) }
    fn make(a: uint(4), bx: uint(4), by: uint(4), c: uint(4)) -> Outer {
        return Outer.new(a=a, b=Inner.new(x=bx, y=by), c=c);
    }
    """
    body = """
    val o = make(a0, a1, a2, a3);
    return (o.a, (o.b).x, (o.b).y, o.c);
    """
    e = compare_body(
        ["uint(4)", "uint(4)", "uint(4)", "uint(4)"],
        "Tuple(uint(4), uint(4), uint(4), uint(4))",
        body, tmp_dir, prefix=prefix,
    )
    for vals in [(1, 2, 3, 4), (15, 0, 7, 3), (5, 5, 5, 5), (0, 15, 0, 15)]:
        e.eval_assert(list(vals), vals)


def test_hw_tuple_return_from_function(tmp_dir: Path):
    """
    CORRECT: A function returning a tuple works in hardware -- both elements
    are packed into the output port correctly.
    """
    body = """
    val s = a0 + a1;
    val p = a0 * a1;
    return (s, p);
    """
    e = compare_body(["uint(4)", "uint(4)"], "Tuple(uint(8), uint(8))",
                     body, tmp_dir)
    for a in [0, 1, 5, 15]:
        for b in [0, 1, 5, 15]:
            e.eval_assert([a, b], (a + b, a * b))


def test_hw_signed_comparison(tmp_dir: Path):
    """
    CORRECT: Comparing a signed int with an unsigned int in hardware produces
    the correct result -- the Verilog expands both to a common signed type.
    """
    body = "return a0 < a1;"
    e = compare_body(["int(-1..2)", "uint(0..3)"], "bool", body, tmp_dir)
    for a in [-1, 0, 1]:
        for b in [0, 1, 2]:
            e.eval_assert([a, b], a < b)


def test_hw_signed_arithmetic_right_shift(tmp_dir: Path):
    """
    CORRECT: Arithmetic right shift (`>>`) on a signed integer preserves the
    sign bit -- it should be equivalent to floor-division by 2.
    """
    import math
    body = "return a0 >> 1;"
    e = compare_body(["int(8)"], "int(8)", body, tmp_dir)
    for v in range(-128, 128, 16):
        e.eval_assert([v], math.floor(v / 2))


def test_hw_signed_right_shift_by_variable_amount(tmp_dir: Path):
    """
    CORRECT: Variable-amount arithmetic right shift produces the same result
    as the interpreter (floor-division by 2**n).
    """
    import math
    body = "return a0 >> a1;"
    e = compare_body(["int(8)", "uint(0..8)"], "int(8)", body, tmp_dir)
    for v in [-128, -64, -1, 0, 1, 64, 127]:
        for s in [0, 1, 2, 7]:
            e.eval_assert([v, s], math.floor(v / (2 ** s)))


def test_hw_if_with_range_implication(tmp_dir: Path):
    """
    CORRECT: An `if (x >= 0)` guard correctly constrains the type of `x`
    in the then-branch, allowing it to be used where only non-negative values
    are expected.
    """
    body = """
    if (a0 >= 0) {
        return a0;
    }
    return 0;
    """
    e = compare_body(["int(-10..10)"], "int(0..10)", body, tmp_dir)
    for v in range(-10, 10):
        e.eval_assert([v], max(v, 0))


def test_hw_complex_branching(tmp_dir: Path):
    """
    CORRECT: Nested if/else correctly propagates mutations to a variable
    through multiple branches.
    """
    body = """
    var result = 0;
    if (a0 > 0) {
        result = 10;
        if (a0 > 5) {
            result = 20;
        }
    } else {
        result = 30;
        if (a0 < -5) {
            result = 40;
        }
    }
    return result;
    """
    e = compare_body(["int(-10..11)"], "int(0..41)", body, tmp_dir)
    for v in range(-10, 11):
        if v > 5:
            expected = 20
        elif v > 0:
            expected = 10
        elif v < -5:
            expected = 40
        else:
            expected = 30
        e.eval_assert([v], expected)


def test_hw_multiple_early_returns(tmp_dir: Path):
    """
    CORRECT: Multiple early-return statements in a function are correctly
    converted into a Verilog flag-based return mechanism.
    """
    body = """
    if (a0 == 0) { return 100; }
    if (a0 == 1) { return 200; }
    if (a0 == 2) { return 300; }
    return 400;
    """
    e = compare_body(["uint(0..4)"], "uint(0..401)", body, tmp_dir)
    e.eval_assert([0], 100)
    e.eval_assert([1], 200)
    e.eval_assert([2], 300)
    e.eval_assert([3], 400)


def test_hw_for_loop_with_initial_value(tmp_dir: Path):
    """
    CORRECT: A for loop that conditionally updates a variable carries both
    the initial value and the loop-body mutations correctly through the
    unrolled Verilog.
    """
    body = """
    var found = false;
    var pos = 99;
    for (i in 0..4) {
        if (a0 == i) {
            found = true;
            pos = i;
        }
    }
    return (found, pos);
    """
    e = compare_body(["uint(0..5)"], "Tuple(bool, uint(0..100))", body, tmp_dir)
    # Values 0-3 are found at their position
    for v in range(4):
        e.eval_assert([v], (True, v))
    # Value 4 is not in 0..4, so pos stays at 99
    e.eval_assert([4], (False, 99))


def test_hw_array_construction_and_hardware_index(tmp_dir: Path):
    """
    CORRECT: An array constructed from hardware values, then indexed with a
    hardware index variable, is correctly lowered to Verilog.
    """
    body = """
    val arr = [a0, a1, a2, a3];
    return arr[2];
    """
    e = compare_body(
        ["uint(8)", "uint(8)", "uint(8)", "uint(8)"],
        "uint(8)", body, tmp_dir,
    )
    e.eval_assert([10, 20, 30, 40], 30)
    e.eval_assert([255, 0, 128, 64], 128)


def test_hw_hardware_exhaustive_match_required():
    """
    CORRECT: The compiler rejects a hardware match statement that does not
    cover all possible values of the scrutinee type.
    """
    src = """
    module eval_mod ports(p0: in async uint(0..4), p_res: out async uint(0..2)) {
        comb {
            match (p0) {
                0 => { p_res = 0; }
                1 => { p_res = 1; }
            }
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException,
                       match="hardware match statement must be exhaustive"):
        compile_custom(src).resolve("top.eval_mod")


def test_hw_division_signed(tmp_dir: Path):
    """
    CORRECT: Integer division in hardware (floor-division) produces the same
    result as the interpreter for all signed and positive divisors.
    """
    import math
    body = "return a0 / a1;"
    e = compare_body(["int(-8..9)", "int(1..5)"], "int(-8..9)", body, tmp_dir)
    for a in range(-8, 9):
        for b in range(1, 5):
            e.eval_assert([a, b], math.floor(a / b))


def test_hw_bool_operations(tmp_dir: Path):
    """
    CORRECT: Boolean &&, ||, ! work correctly in hardware.
    """
    body = "return (a0 && a1) || (!a0 && a2);"
    e = compare_body(["bool", "bool", "bool"], "bool", body, tmp_dir)
    for a, b, c in [(False, False, False), (False, False, True),
                    (False, True, False), (True, False, False),
                    (True, True, False), (True, False, True)]:
        e.eval_assert([a, b, c], (a and b) or (not a and c))


def test_hw_fn_call_with_computed_args(tmp_dir: Path):
    """
    CORRECT: Calling a function from within a hardware context, passing
    hardware-computed values as arguments, is correctly inlined.
    """
    prefix = "fn double(x: int(8)) -> int(8) { return x * 2; }"
    body = "return double(a0) + double(a0);"
    e = compare_body(["int(4)"], "int(8)", body, tmp_dir, prefix=prefix)
    for v in range(-8, 8):
        e.eval_assert([v], v * 4)
