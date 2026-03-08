"""
Tests to find bugs in the HwLang compiler: type system issues, miscompilations,
wrong results, ICEs, and panics.

Each test documents the expected behavior and confirms whether the compiler
handles the case correctly.
"""
from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.compare import compare_body, compare_expression
from hwl_sandbox.common.util import compile_custom


# =============================================================================
# Bug: array[index].field parse error
# =============================================================================

def test_parse_array_index_then_dot_field():
    """
    Chaining array indexing with dot field access fails to parse.
    `a[i].field` gives "unexpected token `.`" while `(a[i]).field` works.
    This is a known TODO in grammar.lalrpop.
    """
    src = """
    struct Point { x: int, y: int }
    fn f() -> int {
        val p1 = Point.new(x=1, y=2);
        val p2 = Point.new(x=3, y=4);
        val pts = [p1, p2];
        return pts[1].x;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_array_index_then_dot_int():
    """
    Accessing a tuple element from an array index: `a[i].0` fails to parse.
    """
    src = """
    fn f() -> int {
        val a = [(1, 2), (3, 4)];
        return a[1].0;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


def test_parse_workaround_array_index_then_dot():
    """Workaround: parenthesising the array index expression allows dot access."""
    src = """
    struct Point { x: int, y: int }
    fn f() -> int {
        val p1 = Point.new(x=1, y=2);
        val p2 = Point.new(x=3, y=4);
        val pts = [p1, p2];
        return (pts[1]).x;
    }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 3


# =============================================================================
# Bug: struct field assignment not implemented
# =============================================================================

def test_struct_field_assignment_not_implemented():
    """
    Assigning to a struct field (`p.x = 5`) is not yet implemented.
    The compiler gives "feature not yet implemented: assignment target dot index
    on non-interface".
    """
    src = """
    struct Pair { x: int, y: int }
    fn f() -> Pair {
        var p = Pair.new(x=1, y=2);
        p.x = 5;
        return p;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="feature not yet implemented"):
        compile_custom(src).resolve("top.f")


# =============================================================================
# Bug: tuple destructuring not implemented
# =============================================================================

def test_tuple_destructuring_not_implemented():
    """
    Tuple destructuring in val declarations (`val (n, b) = t;`) is not supported.
    """
    src = """
    fn f(t: Tuple(int, bool)) -> int {
        val (n, b) = t;
        return n + bool_to_int(b);
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token"):
        compile_custom(src).resolve("top.f")


# =============================================================================
# Correctness: division with signed negative operands in hardware
# =============================================================================

def test_hardware_div_signed_negative_dividend(tmp_dir: Path):
    """
    Floor division in hardware with negative dividend should match Python semantics.
    """
    e = compare_expression(["int(-8..1)", "int(1..4)"], "int(-8..=1)", "a0 / a1", tmp_dir)
    for x in range(-8, 1):
        for y in [1, 2, 3]:
            import math
            e.eval_assert([x, y], math.floor(x / y))


def test_hardware_mod_signed_negative_dividend(tmp_dir: Path):
    """
    Modulo in hardware with negative dividend (floor modulo) should match Python semantics.
    """
    e = compare_expression(["int(-8..1)", "int(1..4)"], "int(0..4)", "a0 % a1", tmp_dir)
    for x in range(-8, 1):
        for y in [1, 2, 3]:
            e.eval_assert([x, y], x % y)


# =============================================================================
# Correctness: bit operations in hardware
# =============================================================================

def test_hardware_bool_and(tmp_dir: Path):
    """Boolean AND in hardware: both function and Verilog should agree."""
    e = compare_expression(["bool", "bool"], "bool", "a0 && a1", tmp_dir)
    for a in [False, True]:
        for b in [False, True]:
            e.eval_assert([a, b], a and b)


def test_hardware_bool_or(tmp_dir: Path):
    """Boolean OR in hardware."""
    e = compare_expression(["bool", "bool"], "bool", "a0 || a1", tmp_dir)
    for a in [False, True]:
        for b in [False, True]:
            e.eval_assert([a, b], a or b)


def test_hardware_bool_xor(tmp_dir: Path):
    """Boolean XOR in hardware."""
    e = compare_expression(["bool", "bool"], "bool", "a0 ^^ a1", tmp_dir)
    for a in [False, True]:
        for b in [False, True]:
            e.eval_assert([a, b], a ^ b)


def test_hardware_bool_not(tmp_dir: Path):
    """Boolean NOT in hardware: both function and Verilog should agree."""
    e = compare_expression(["bool"], "bool", "!a0", tmp_dir)
    e.eval_assert([False], True)
    e.eval_assert([True], False)


# =============================================================================
# Correctness: signed shift operations in hardware
# =============================================================================

def test_hardware_arithmetic_right_shift(tmp_dir: Path):
    """Arithmetic right shift (>>) on signed integers should preserve sign bit."""
    e = compare_expression(["int(8)", "uint(0..8)"], "int(8)", "a0 >> a1", tmp_dir)
    for x in [-128, -64, -1, 0, 1, 64, 127]:
        for shift in [0, 1, 3, 7]:
            import math
            expected = math.floor(x / (2 ** shift))
            e.eval_assert([x, shift], expected)


def test_hardware_left_shift_signed(tmp_dir: Path):
    """Left shift on signed integers should work correctly."""
    e = compare_expression(["int(-8..8)", "uint(0..4)"], "int(-128..128)", "a0 << a1", tmp_dir)
    for x in [-7, -1, 0, 1, 7]:
        for shift in [0, 1, 2, 3]:
            e.eval_assert([x, shift], x * (2 ** shift))


# =============================================================================
# Correctness: match in hardware with enum
# =============================================================================

def test_hardware_enum_match(tmp_dir: Path):
    """Enum construction and match in hardware should work correctly."""
    prefix = """
    enum Dir { N, S, E, W }
    """
    body = """
    match (a0) {
        .N => { return 0; }
        .S => { return 1; }
        .E => { return 2; }
        .W => { return 3; }
    }
    """
    e = compare_body(["Dir"], "uint(0..4)", body, tmp_dir, prefix=prefix)

    src_make = """
    enum Dir { N, S, E, W }
    fn n() -> Dir { return Dir.N; }
    fn s() -> Dir { return Dir.S; }
    fn east() -> Dir { return Dir.E; }
    fn w() -> Dir { return Dir.W; }
    """
    c = compile_custom(src_make)
    e.eval_assert([c.resolve("top.n")()], 0)
    e.eval_assert([c.resolve("top.s")()], 1)
    e.eval_assert([c.resolve("top.east")()], 2)
    e.eval_assert([c.resolve("top.w")()], 3)


def test_hardware_enum_with_payload_match(tmp_dir: Path):
    """Enum with payload in hardware match should extract payload correctly."""
    prefix = """
    enum Msg { A(uint(4)), B(uint(4)), C }
    """
    body = """
    match (a0) {
        .A(val v) => { return (0, v); }
        .B(val v) => { return (1, v); }
        .C => { return (2, 0); }
    }
    """
    e = compare_body(["Msg"], "Tuple(uint(0..3), uint(0..16))", body, tmp_dir, prefix=prefix)

    src_make = """
    enum Msg { A(uint(4)), B(uint(4)), C }
    fn make_a(v: uint(4)) -> Msg { return Msg.A(v); }
    fn make_b(v: uint(4)) -> Msg { return Msg.B(v); }
    fn make_c() -> Msg { return Msg.C; }
    """
    c = compile_custom(src_make)
    make_a = c.resolve("top.make_a")
    make_b = c.resolve("top.make_b")
    make_c = c.resolve("top.make_c")

    e.eval_assert([make_a(5)], (0, 5))
    e.eval_assert([make_a(15)], (0, 15))
    e.eval_assert([make_b(3)], (1, 3))
    e.eval_assert([make_c()], (2, 0))


# =============================================================================
# Correctness: for loop unrolling in hardware
# =============================================================================

def test_hardware_for_loop_sum(tmp_dir: Path):
    """For loop unrolling in hardware: sum of 0..8 should be constant 28."""
    body = """
    var sum = 0;
    for (i in 0..8) {
        sum = sum + i;
    }
    return sum;
    """
    e = compare_body([], "int(0..=28)", body, tmp_dir)
    e.eval_assert([], 28)


def test_hardware_for_loop_with_input(tmp_dir: Path):
    """For loop unrolling: multiply an input by a loop-produced constant."""
    body = """
    var factor = 1;
    for (i in 0..4) {
        factor = factor * 2;
    }
    return a0 * factor;
    """
    e = compare_body(["uint(8)"], "int(0..4096)", body, tmp_dir)
    for v in [0, 1, 5, 10, 255]:
        e.eval_assert([v], v * 16)


# =============================================================================
# Correctness: struct in hardware
# =============================================================================

def test_hardware_struct_fields(tmp_dir: Path):
    """Struct field access in hardware should work correctly."""
    prefix = """
    struct RGB { r: uint(4), g: uint(4), b: uint(4) }
    """
    body = """
    val c = RGB.new(r=a0, g=a1, b=a2);
    return (c.r, c.g, c.b);
    """
    e = compare_body(
        ["uint(4)", "uint(4)", "uint(4)"],
        "Tuple(uint(4), uint(4), uint(4))",
        body, tmp_dir, prefix=prefix
    )
    e.eval_assert([1, 2, 3], (1, 2, 3))
    e.eval_assert([15, 0, 7], (15, 0, 7))
    e.eval_assert([0, 0, 0], (0, 0, 0))


# =============================================================================
# Correctness: register behavior
# =============================================================================

def test_register_counter(tmp_dir: Path):
    """A simple counter register should increment each clock cycle."""
    src = """
    module counter ports(
        clk: in clock,
        rst: in async bool,
        y: out sync(clk, async rst) uint(8)
    ) {
        clocked(clk, async rst) {
            reg wire y = 0;
            y = y + 1;
        }
    }
    """
    c = compile_custom(src)
    m = c.resolve("top.counter")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    inst = m.as_verilated(tmp_dir).instance()

    # Reset
    inst.ports.rst.value = True
    inst.step(1)
    assert inst.ports.y.value == 0

    inst.ports.rst.value = False
    inst.step(1)
    assert inst.ports.y.value == 1
    inst.step(1)
    assert inst.ports.y.value == 2
    inst.step(1)
    assert inst.ports.y.value == 3


def test_register_pipeline(tmp_dir: Path):
    """A two-stage pipeline register should delay input by 2 cycles."""
    src = """
    module pipe2 ports(
        clk: in clock,
        rst: in async bool,
        x: in sync(clk, async rst) uint(8),
        y: out sync(clk, async rst) uint(8)
    ) {
        wire stage1: uint(8);
        clocked(clk, async rst) {
            reg wire stage1 = 0;
            reg wire y = 0;
            stage1 = x;
            y = stage1;
        }
    }
    """
    c = compile_custom(src)
    m = c.resolve("top.pipe2")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    inst = m.as_verilated(tmp_dir).instance()

    inst.ports.rst.value = True
    inst.step(1)
    inst.ports.rst.value = False

    # Cycle 1: input = 42
    inst.ports.x.value = 42
    inst.step(1)
    # y is still 0 (2-stage delay)

    # Cycle 2: input = 99
    inst.ports.x.value = 99
    inst.step(1)
    assert inst.ports.y.value == 42  # first value arrives

    # Cycle 3
    inst.ports.x.value = 0
    inst.step(1)
    assert inst.ports.y.value == 99  # second value arrives


# =============================================================================
# Correctness: module instantiation
# =============================================================================

def test_module_instance_connections(tmp_dir: Path):
    """Module instantiation with correct port connections should work."""
    src = """
    module adder ports(a: in async uint(8), b: in async uint(8), c: out async uint(9)) {
        comb { c = a + b; }
    }
    module top ports(x: in async uint(8), y: in async uint(8), z: out async uint(9)) {
        instance adder ports(a=x, b=y, c=z);
    }
    """
    c = compile_custom(src)
    m = c.resolve("top.top")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    inst = m.as_verilated(tmp_dir).instance()

    inst.ports.x.value = 10
    inst.ports.y.value = 20
    inst.step(1)
    assert inst.ports.z.value == 30

    inst.ports.x.value = 200
    inst.ports.y.value = 200
    inst.step(1)
    assert inst.ports.z.value == 400


# =============================================================================
# Correctness: comparison operations in hardware
# =============================================================================

def test_hardware_signed_comparison(tmp_dir: Path):
    """Signed integer comparison in hardware must correctly handle negative numbers."""
    e = compare_expression(["int(8)", "int(8)"], "bool", "a0 < a1", tmp_dir)
    cases = [
        (-128, -127), (-1, 0), (-1, 1), (0, 1), (126, 127),
        (0, -1), (1, -1), (127, -128),
    ]
    for a, b in cases:
        e.eval_assert([a, b], a < b)


def test_hardware_unsigned_comparison(tmp_dir: Path):
    """Unsigned integer comparison in hardware."""
    e = compare_expression(["uint(8)", "uint(8)"], "bool", "a0 < a1", tmp_dir)
    cases = [(0, 1), (100, 200), (254, 255), (255, 0), (200, 100)]
    for a, b in cases:
        e.eval_assert([a, b], a < b)


# =============================================================================
# Correctness: array slice in hardware
# =============================================================================

def test_hardware_array_slice(tmp_dir: Path):
    """Array slicing in hardware should produce correct subarrays."""
    body = """
    return a0[1..5];
    """
    e = compare_body(["[8]uint(4)"], "[4]uint(4)", body, tmp_dir)
    arr = [0, 1, 2, 3, 4, 5, 6, 7]
    e.eval_assert([arr], arr[1:5])
    arr2 = [15, 14, 13, 12, 11, 10, 9, 8]
    e.eval_assert([arr2], arr2[1:5])


# =============================================================================
# Correctness: string interpolation
# =============================================================================

def test_string_interpolation_basic():
    """String interpolation with integers should produce correct strings."""
    src = """
    fn f(n: int, m: int) -> str {
        return "n={n}, m={m}";
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f(42, -7) == "n=42, m=-7"
    assert f(0, 0) == "n=0, m=0"


# =============================================================================
# Correctness: bool_to_int and int operations
# =============================================================================

def test_bool_to_int(tmp_dir: Path):
    """bool_to_int should convert False->0, True->1 in both function and hardware."""
    e = compare_body(["bool"], "uint(0..2)", "return bool_to_int(a0);", tmp_dir)
    e.eval_assert([False], 0)
    e.eval_assert([True], 1)


# =============================================================================
# Type system: overflow detection
# =============================================================================

def test_type_out_of_range_assignment():
    """Assigning a value outside the declared range should give a type error."""
    src = """
    fn f() -> uint(8) {
        return 256;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        compile_custom(src).resolve("top.f")


def test_type_negative_in_uint():
    """Assigning a negative value to a uint type should give a type error."""
    src = """
    fn f() -> uint(8) {
        return -1;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        compile_custom(src).resolve("top.f")


# =============================================================================
# Type system: power operator edge cases
# =============================================================================

def test_zero_pow_zero_error():
    """0**0 should give a compile error since both operands could be 0."""
    src = "fn f() -> int { return 0 ** 0; }"
    with pytest.raises(hwl.DiagnosticException, match="invalid power operation"):
        compile_custom(src).resolve("top.f")


def test_int_x_pow_zero_error():
    """x**0 where x: int (can be 0) should give a compile error."""
    src = "fn f(x: int) -> int { return x ** 0; }"
    with pytest.raises(hwl.DiagnosticException, match="invalid power operation"):
        compile_custom(src).resolve("top.f")


def test_natural_x_pow_zero_ok():
    """x**0 where x: natural (always > 0) should compile and return 1."""
    src = "fn f(x: natural) -> uint { return x ** 0; }"
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f(1) == 1
    assert f(5) == 1
    assert f(100) == 1


# =============================================================================
# Error handling: missing return paths
# =============================================================================

def test_missing_return_detected():
    """A function missing a return path should give a diagnostic error."""
    src = """
    fn f(x: bool) -> int {
        if (x) {
            return 1;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="missing return"):
        compile_custom(src).resolve("top.f")


def test_missing_return_partial_is_runtime_error():
    """
    A function with some (but not all) return paths covered compiles OK,
    but hitting the unhandled path at runtime gives a diagnostic error.
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


# =============================================================================
# Error handling: port direction
# =============================================================================

def test_cannot_assign_to_input_port():
    """Assigning to an input port should give a compile-time error."""
    src = """
    module top ports(x: in async bool, y: in async bool) {
        comb { x = y; }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="cannot assign to input port"):
        compile_custom(src).resolve("top.top")


# =============================================================================
# Correctness: Option and Result enum variants
# =============================================================================

def test_option_none_is_not_callable():
    """Option.None is a value, not a function - calling it should error."""
    src = """
    fn f() {
        val v: Option(bool) = Option(bool).None();
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="call target must be function"):
        compile_custom(src).resolve("top.f")


def test_option_some_and_none(tmp_dir: Path):
    """Option enum should correctly represent Some and None values in hardware."""
    prefix = "// using built-in Option"
    body = """
    match (a0) {
        .None => { return false; }
        .Some(val v) => { return v; }
    }
    """
    e = compare_body(["Option(bool)"], "bool", body, tmp_dir, prefix=prefix)

    src_make = """
    fn make_none() -> Option(bool) { return Option(bool).None; }
    fn make_some(v: bool) -> Option(bool) { return Option(bool).Some(v); }
    """
    c = compile_custom(src_make)
    none = c.resolve("top.make_none")()
    some_true = c.resolve("top.make_some")(True)
    some_false = c.resolve("top.make_some")(False)

    e.eval_assert([none], False)
    e.eval_assert([some_true], True)
    e.eval_assert([some_false], False)


# =============================================================================
# Correctness: dynamic IDs and for loop with pub wires
# =============================================================================

def test_dynamic_id_shift_register(tmp_dir: Path):
    """Dynamic IDs with pub wires in a for loop should create a shift register."""
    src = """
    module top ports(
        clk: in clock,
        rst: in async bool,
        sync(clk, async rst) {
            x: in int(8),
            y: out int(8),
        }
    ) {
        for (i in 0..4) {
            pub wire id_from_str("w{i}"): int(8);
            comb {
                if (i == 0) {
                    id_from_str("w{i}") = x;
                } else {
                    id_from_str("w{i}") = id_from_str("w{i-1}");
                }
            }
        }

        comb {
            y = w3;
        }
    }
    """
    c = compile_custom(src)
    m = c.resolve("top.top")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    inst = m.as_verilated(tmp_dir).instance()

    inst.ports.x.value = 4
    inst.step(1)
    assert inst.ports.y.value == 4


# =============================================================================
# Correctness: large integer hardware operations
# =============================================================================

def test_hardware_large_int_add(tmp_dir: Path):
    """Addition of large integers (> 64-bit) should be handled correctly in hardware."""
    e = compare_expression(
        ["int(0..2**128)", "int(0..=0)"],
        "int(0..2**128)",
        "a0 + a1",
        tmp_dir
    )
    e.eval_assert([0, 0], 0)
    e.eval_assert([2 ** 64 - 1, 0], 2 ** 64 - 1)
    e.eval_assert([2 ** 65 - 1, 0], 2 ** 65 - 1)


# =============================================================================
# Correctness: recursive functions
# =============================================================================

def test_recursive_function():
    """Recursive functions should evaluate correctly at compile time."""
    src = """
    fn factorial(n: uint) -> uint {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.factorial")
    assert f(0) == 1
    assert f(1) == 1
    assert f(5) == 120
    assert f(10) == 3628800


# =============================================================================
# Correctness: closures and higher-order functions
# =============================================================================

def test_closure_capture():
    """Closures should correctly capture outer variables."""
    src = """
    fn make_adder(n: int) -> Function {
        fn adder(x: int) -> int {
            return x + n;
        }
        return adder;
    }
    fn test() -> int {
        val add5 = make_adder(5);
        val add10 = make_adder(10);
        return add5(3) + add10(7);
    }
    """
    c = compile_custom(src)
    test = c.resolve("top.test")
    assert test() == 25  # (3+5) + (7+10)


# =============================================================================
# Correctness: array comprehension
# =============================================================================

def test_array_comprehension(tmp_dir: Path):
    """Array comprehension should produce correct results in both function and hardware."""
    body = "return [i * i for i in 0..a0];"
    # For N=5: [0, 1, 4, 9, 16]
    e = compare_body(["uint(0..=5)"], "[5]uint(0..=16)", body, tmp_dir)
    e.eval_assert([5], [0, 1, 4, 9, 16])


# =============================================================================
# Type system: forward references
# =============================================================================

def test_forward_reference_function():
    """Functions should be callable before their definition in the file."""
    src = """
    fn f() -> int { return g(); }
    fn g() -> int { return 42; }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f() == 42


# =============================================================================
# Correctness: loop control flow
# =============================================================================

def test_break_in_for_loop():
    """Break should exit the loop early."""
    src = """
    fn f() -> int {
        var count = 0;
        for (i in 0..100) {
            if (i == 5) { break; }
            count = count + 1;
        }
        return count;
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f() == 5


def test_continue_in_for_loop():
    """Continue should skip the rest of the loop body."""
    src = """
    fn f() -> int {
        var sum = 0;
        for (i in 0..10) {
            if (i % 2 == 0) { continue; }
            sum = sum + i;
        }
        return sum;
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f() == 25  # 1+3+5+7+9


# =============================================================================
# Correctness: ref/deref
# =============================================================================

def test_ref_deref_basic():
    """ref/deref should allow indirect mutation."""
    src = """
    fn f() -> int {
        var x = 0;
        val r = ref(x);
        deref(r) = 42;
        return x;
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f() == 42


# =============================================================================
# Error: undriven output port
# =============================================================================

def test_undriven_output_port_warning():
    """An output port with no driver should emit a warning."""
    src = """
    module top ports(x: out async bool) {}
    """
    with pytest.raises(hwl.DiagnosticException, match="port.*has no driver"):
        compile_custom(src).resolve("top.top")


# =============================================================================
# Correctness: empty tuple and single-element tuple
# =============================================================================

def test_empty_tuple(tmp_dir: Path):
    """Empty tuples should work in both function and hardware contexts."""
    e = compare_expression([], "Tuple()", "()", tmp_dir)
    e.eval_assert([], ())


def test_single_element_tuple(tmp_dir: Path):
    """Single-element tuples should be correctly handled."""
    e = compare_expression(["uint(8)"], "Tuple(uint(8),)", "(a0,)", tmp_dir)
    e.eval_assert([0], (0,))
    e.eval_assert([42], (42,))


# =============================================================================
# Correctness: interface with parameters
# =============================================================================

def test_parameterized_interface(tmp_dir: Path):
    """Parameterized interfaces should correctly pass data."""
    src = """
    interface Bus(W: uint) {
        data: uint(W),
        interface input { data: in }
        interface output { data: out }
    }
    module top ports(
        x: interface async Bus(8).input,
        y: interface async Bus(8).output
    ) {
        comb { y.data = x.data; }
    }
    """
    c = compile_custom(src)
    m = c.resolve("top.top")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    inst = m.as_verilated(tmp_dir).instance()

    inst.ports.x_data.value = 123
    inst.step(1)
    assert inst.ports.y_data.value == 123


# =============================================================================
# Correctness: var with multiple assignments
# =============================================================================

def test_var_multiple_assignments():
    """Multiple assignments to the same var variable should use the last value."""
    src = """
    fn f(x: bool) -> bool {
        var y = false;
        y = x;
        y = !x;
        return y;
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")
    assert f(True) == False   # last write is !True = False
    assert f(False) == True   # last write is !False = True


def test_var_multiple_assignments_in_hardware(tmp_dir: Path):
    """Multiple assignments to a var in hardware should use the last value."""
    body = """
    var y = false;
    y = a0;
    y = !a0;
    return y;
    """
    e = compare_body(["bool"], "bool", body, tmp_dir)
    e.eval_assert([True], False)   # last write is !True = False
    e.eval_assert([False], True)   # last write is !False = True
