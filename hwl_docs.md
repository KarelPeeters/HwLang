# HWLang RTL Language Reference

## Overview

HWLang is an experimental hardware description language that compiles to Verilog 2005.
Source files use the `.kh` extension. Projects are defined via `hwl.toml` manifests.

## Basic Syntax

### Modules

```hwl
// Internal module (with body)
pub module module_name(
    generic_param: type,
    N: uint,
) ports(
    clk: in clock,
    rst: in async bool,
    sync(clk, async rst) {
        input_data: in bool,
        output_data: out uint(8),
    }
) {
    // body
}

// External module (Verilog interop, no body)
pub external module ram_single_port(W: uint, H: uint) ports(
    clk: in clock,
    address: in uint(W),
    write_data: in [W]bool,
    read_data: out [W]bool,
)
```

### Ports

Directions: `in`, `out`. Domains: `async`, `sync(clk)`, `sync(clk, rst)`.
Interface ports: `stream: interface async axi_stream(T).input`.

### Types

- **bool**: 1-bit boolean (`true`, `false`)
- **int**: Infinite-precision signed integer (compile-time)
- **uint** = `int(0..)`: Unsigned integer
- **int(N)**: Signed N-bit integer, range `int(-2**(N-1)..2**(N-1))`
- **uint(N)**: Unsigned N-bit, range `int(0..2**N)`
- **Ranges**: `int(0..8)` (exclusive), `int(0..=8)` (inclusive), `int(0..)` (unbounded)
- **Arrays**: `[N]T` (fixed size), `[_]T` (inferred size)
- **Tuples**: `Tuple(T1, T2, ...)`, indexed as `t.0`, `t.1`, ...
- **Structs**: `struct Foo(T: type) { x: T, y: T }`, constructed via `Foo(bool).new(x=true, y=false)`, accessed via `s.x`
- **Enums**: `enum Option(T: type) { None, Some(T) }`, constructed via `Option(bool).None`
- **Interfaces**: Bundles of signals with directional views

### Generics

Parameters can be types, values, functions, or modules:

```hwl
module child(T: type, N: uint) ports(...) { ... }
instance child(T=bool, N=4) ports(...);
```

Can have defaults and conditional inclusion (`if condition { param: type }`).

### Functions

```hwl
fn select(T: type, c: bool, a: T, b: T) -> T {
    if (c) { return a; } else { return b; }
}
```

Functions can be pure (compile-time) or combinational. Evaluated at compile time when all args are const.

### Variables

| Keyword | Mutable | Scope    | Hardware |
|---------|---------|----------|----------|
| `const` | No      | Global   | No       |
| `val`   | No      | Local    | Optional |
| `var`   | Yes     | Local    | Optional |
| `wire`  | Yes     | Module   | Yes      |
| `reg`   | Yes     | Module   | Yes      |

### Processes

```hwl
// Combinatorial
comb {
    output.data = input.data;
}

// Clocked with async reset
clocked(clk, async rst) {
    reg wire counter = 0;
    counter += 1;
}
```

### Control Flow

- **if/else**: `if (cond) { ... } else { ... }`
- **match**: `match (v) { 0 => { ... } .Variant(val p) => { ... } }`
- **for**: `for (i in 0..N) { ... }` (unrolled in hardware)
- **while/loop**: `while (cond) { ... }`, `loop { ... }`
- **break/continue/return**

### Expressions

Arithmetic: `+ - * / % **`
Bitwise: `& | ^ ^^ !`
Logical: `&& ||`
Comparison: `== != < <= > >=`
Shift: `<< >>`

### Arrays

- Literal: `[1, 2, 3]`
- Repeat: `[false] * N`
- Comprehension: `[expr for i in 0..N]`
- Index: `arr[i]`
- Slice: `arr[start..end]`, `arr[..end]`, `arr[start..]`
- Spread: `[*arr, new_element]`
- Length: `arr.len`

### Bit Manipulation

- `type.to_bits(value)` → `[N]bool`
- `type.from_bits(bits)` → value (safe)
- `type.from_bits_unsafe(bits)` → value (unsafe)
- `type.size_bits` → compile-time bit width

### Module Instantiation

```hwl
instance child_module(T=bool, N=4) ports(
    clk=clk,
    input_data=data,
    output_data=result,
);
```

Port name shorthand: just use the name when it matches the variable name.

### Domain System

Every value has a domain: `const`, `async`, `sync(clk)`, `sync(clk, rst)`.
Crossing domains requires explicit `unsafe_value_with_domain(value, domain)`.

### Misc

- `undef`: uninitialized/undefined value
- `ref(v)`, `deref(r)`: reference and dereference signals
- `id_from_str("v_{i}")`: dynamic identifier generation
- `print(...)`, `assert(...)`: compile-time debugging
