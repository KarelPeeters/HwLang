# KHDL: research summary (source material for lyrics)

KHDL is an experimental hardware description language by Karel Peeters. The source files use the `.kh` extension, and the
project manifest is `khdl.toml`. The project was first named "HwLang" (`hwl`), then "KH", and was renamed KHDL in
October 2026, partly because an LLM spontaneously invented ```` ```kh ```` code fences and "now kh is in the training data".
The sites are khdl-lang.org and khdl-lang.com, the GitHub org is `github.com/khdl-lang`, and khdl-lang.org hosts a WASM
playground. The first commit ("Empty rust project.") is from July 2023. The repository has about 1480 commits and the
compiler is about 50k lines of Rust.

As instructed, this summary treats every planned feature and bugfix described anywhere in the repository as already
implemented. Section 5 still notes which items started out as plans, because "it's on the roadmap" can be lyric material
too.

---

## 1. Elevator pitch and philosophy

**Pitch.** KHDL is a modern, strongly typed RTL language that compiles to plain, lowest-common-denominator Verilog 2005,
so it works with every EDA tool you already own. It tracks integer *ranges* instead of bit widths, so overflow and
truncation are compile errors rather than silicon bugs. It tracks *clock and reset domains* on every port, wire and
value, so CDC mistakes are caught on every compile instead of in a sign-off linter three months later. There is exactly
one assignment operator, `=`, with software semantics: no blocking/non-blocking split and no sensitivity lists. Registers,
wires and variables are separate keywords, so you cannot accidentally infer a latch or a register. Generics are ordinary
compile-time code. Types, modules, functions, interfaces and even packages are first-class values, and a Zig-like
compile-time interpreter replaces `generate` blocks, Python codegen scripts and editor templates. It ships with Rust-grade
error messages, a built-in LSP, an auto-formatter, Python bindings with Verilator simulation, and a multithreaded compiler.

**Core strategy: "decouple language from tooling".** SystemVerilog and VHDL can't improve quickly. Every new feature has
to be supported by every simulator, synthesizer, linter, LEC tool, power estimator and IDE from different vendors that
share no front-ends, which takes years and many support tickets. KHDL compiles a new language to a simple Verilog subset
instead: "If we find that certain constructs are not well supported by any EDA tool, we can change the Verilog code
generation to fix it." The language can evolve without waiting on Synopsys, Cadence or Siemens. Its model is Kotlin, "a
second generation language that replaces Java while still having very good interop."

**Design guidelines** (quoted or paraphrased from `design/design.md`):
- Modules, functions and structs typecheck **on their own, independent of their users**: "an error is _either_ inside of
  something or at the use site. It can't propagate outwards. Unlike C++ template parameters and VHDL functions and
  modules!"
- "Put enough semantics into the language to avoid any possible backend errors." Clocks are marked as clocks and values
  know which clock they are synced to. **"No delta cycles!"**
- "Don't raise the abstraction level too much. The goal is not to invent something completely new, just to drastically
  improve the ergonomics." KHDL is RTL, not HLS.
- Prefer the stdlib over language features, and give the stdlib as little special status as possible.
- Interoperate with VHDL, Verilog and SystemVerilog as much as possible.
- "We don't want _any_ implicit type conversion at all."
- "We want to avoid any 'unpredictable' or undefined behavior."
- Inspirations: Python's convenient infinite `int`; Rust's safety and type system ("preventing as many bugs as possible
  by making the compiler check for them continuously during development") and cargo; Zig's compile-time execution;
  Kotlin's interop story; Jan Decaluwe's essays on counting and signal assignments; and matklad/rust-analyzer for compiler
  and IDE architecture.
- Author's style advice to users (and to LLMs writing KHDL): **"this is a modern language, so don't constantly make
  defensive design choices that come from PTSD with existing RTL languages."**

---

## 2. Language features, with real syntax

### 2.1 Keywords (exact list from the tokenizer)

`import type struct enum self ports port module interface view instance fn comb clock clocked const val var wire reg ref
deref in out async sync return break continue true false undef if else loop match for while pub as external`, plus the
builtins `__builtin`, `unsafe_value_with_domain` and `ident`.

Operators include `..` (exclusive range), `..=` (inclusive range), `+..` (start plus length, as in `buffer[level+..input_width]`),
`&& || ^^` (boolean and/or/**xor**), `& | ^` (bitwise), `<< >>`, `+ - * / % **`, **`+/` (ceil-division operator)**,
`!`, `=>`, `::`, `_` (dummy/wildcard), `*` (spread in array literals), and the compound assignments `+= -= *= /= %= &= |= ^=`.

The precedence rules are **deliberately strict** about mixing:
- `&&`, `||` and `^^` cannot be mixed without parentheses ("assoc left, no mixing"), and the same goes for `&`, `|` and `^`.
- Comparisons are non-associative, so `a < b < c` is not allowed. Shifts and `**` are also non-associative.

### 2.2 A whole module (the README example)

```kh
pub module top ports(
    clk: in clock,
    rst: in async bool,

    sync(clk, async rst) {
        data_to_add: in uint(0..8),
        stream_in: interface axi_stream(uint(0..8)).input,
        stream_out: interface axi_stream(uint(0..16)).output,
        values_transferred: out uint(32),
    }
) {
    comb {
        stream_in.ready = stream_out.ready;
        stream_out.valid = stream_in.valid;
        stream_out.data = stream_in.data + data_to_add;
    }

    clocked(clk, async rst) {
        reg wire values_transferred = 0;
        if (stream_out.valid && stream_out.ready) {
            values_transferred = (values_transferred + 1) % 2**32;
        }
    }
}

interface axi_stream(T: type) {
    ready: bool, valid: bool, data: T,
    view input { ready: out, valid: in, data: in }
    view output { ready: in, valid: out, data: out }
}
```

Things to notice:
- `stream_in.data + data_to_add` has type `uint(0..15)`, which fits in `uint(0..16)`. **No widths to compute by hand and
  no truncation.**
- The counter has to be wrapped explicitly with `% 2**32`. Writing `values_transferred + 1` alone into `uint(32)` is a
  compile-time type error, because its range is `0..2**32+1`.
- `sync(clk, async rst) { ... }` puts a whole group of ports in one domain.
- `comb { }` and `clocked(clk, async rst) { }` are the only two kinds of processes.

### 2.3 Modules, ports, instances

- A module has the shape `module name(generic params) ports(...) { body }`. Non-generic modules leave out the parameter
  list (`module top ports(...)`). `pub` exports a module.
- Port directions are only `in` and `out`. There is no `inout`.
- A port is declared as `name: in|out <domain> <type>`, as `name: in clock`, or as
  `name: interface <domain> Iface(...).view`.
- Domain blocks group ports: `sync(clk, async rst) { a: in bool, b: out uint(8), }` or `async { ... }`.
- Ports can be **conditional** and **generated by loops**, because all lists in KHDL are "extra lists" that accept `if`
  and `for`:

  ```kh
  module top(c: bool) ports(
      x0: in async bool,
      if (c) { y0: in async bool, }
      async { for (i in 0..8) { ident("x_{i}"): in int(8), ident("y_{i}"): out int(8), } }
  ) { ... }
  ```

- Instantiation uses the `instance` keyword and named connections. When the names match, `ports(clk, rst, x=d1, y)` lets
  you write `clk` instead of `clk=clk`:

  ```kh
  instance child(T=int(0..256)) ports(clk, rst, x=d1, y);
  instance cdc_sync_ff(N) ports(in_data=w_true, out_clk=out_clk, out_rst=in_reset, out_data=out_reset);
  ```

  `_` is a dummy connection for outputs you want to ignore.
- **External modules** wrap existing Verilog/VHDL IP, with generics and ports but no body:

  ```kh
  pub external module ram_single_port_sync(W: uint, H: uint) ports(
      clk: in clock,
      rst: in async bool,
      sync(clk, async rst) {
          address: in int(0..H),
          write_data: in [W]bool,
          write_enable: in bool,
          read_data: out [W]bool,
      }
  )
  ```

- **Generic modules are functions that return modules.** A non-generic module *is* a module value, and a generic one is
  called like a function: `axi_gearbox(T=bool, input_width=8, output_width=3)`. Modules are expressions. "Generic
  arguments are not a property of the instantiation (like VHDL/Verilog/C++), but just normal expressions, i.e. function
  calls." Testers found this surprising, and the author is keeping it ("the advantages are too large").
- Cyclic module instantiation is reported as an error (`cyclic module instantiation`).

### 2.4 Clocks, resets, and domains (the headline feature)

Every value has a **domain**:
- `clock` is a clock signal, a separate kind of thing.
- `async` means "not synchronized to anything".
- `sync(clk)` or `sync(clk, async rst)` means synchronous to a clock, optionally with a reset.
- `const` means a constant hardware value. Compile-time values also have their own domain.

The rules come from `check_valid_domain_crossing`:
- clock to clock is fine. Anything else into a clock is `non-clock to clock`.
- clock into sync is `clock to sync`. **async into sync is `async to sync`**.
- Compile-time and const values can go anywhere.
- Anything can go into `async`.
- Sync to sync needs the same clock **and** a compatible reset. The failures are `different clock`, `different reset`
  and `different clock and reset`. The reset check exists "to protect against the source resetting while the target is
  not."

The error message, shown in `docs/error.png`:

```
error: invalid domain crossing: different clock
  --> top.kh:42:16
   |     x_data: in sync(x_clk) bool,
   |     y_data: out sync(y_clk) bool,
   |                 ----------- info: target domain is sync(y_clk)
   |     comb {
   |         y_data = x_data;
   |                  ^ ------ info: source domain is sync(x_clk)
   |                  invalid domain crossing here
   = info: crossing due to value to target in combinatorial block
   = help: to intentionally cross domains, use `unsafe_value_with_domain` or `unsafe_bool_to_clock`
```

Domains are also checked across control flow. If you branch on an async signal inside a clocked block, the error is
"condition to target", and reading an async reset in a sync-only block produces "invalid domain crossing: async to sync".

**The escape hatch is explicitly unsafe and easy to grep for:** `unsafe_value_with_domain(value, sync(clk))` and
`unsafe_bool_to_clock(b)`, in the spirit of Rust's `unsafe`. The CDC synchronizer example (`examples/cdc.kh`, based on
ZipCPU's CDC posts) is the one place that legitimately uses it:

```kh
module cdc_sync_ff(N: int(2..) = 2) ports(
    in_data: in async bool,
    out_clk: in clock,
    out_rst: in async bool,
    out_data: out sync(out_clk, async out_rst) bool,
) {
    wire buffer: [N]bool;
    clocked(out_clk, async out_rst) {
        reg wire buffer = [false] * N;
        val in_data_domain = unsafe_value_with_domain(in_data, sync(out_clk, async out_rst));
        buffer = [*buffer[1..], in_data_domain];
    }
    comb { out_data = buffer[0]; }
}
```

Note that `N: int(2..)` makes **a 1-flop synchronizer a type error.**

**Resets:**
- `clocked(clk, async rst) { ... }`, `clocked(clk, sync rst) { ... }`, or `clocked(clk) { ... }` with no reset.
- Reset values are declared together with the register: `reg r: T = <reset value>;`.
- **Every register must have a reset value, or be explicitly marked uninitialized with `= undef`.** There is no silent
  "forgot to reset it."
- A clocked block without a reset cannot drive a register that has a reset value. The error is "clocked block without
  reset cannot drive register with reset value", with the hint "either add a reset to the block or use `undef` as the
  reset value".
- During reset, non-reset (`undef`) registers are assigned X ("behavior is implementation-dependent"), so simulations
  that re-run scenarios after a reset still catch use-of-uninitialized bugs. The **reset style is a backend option**:
  reset-dominates-everything (`if rst ... else ...`) versus reset-applied-last-only-to-reset-registers. This matters
  because, as Maarten pointed out, the "else" style adds `!reset` to the clock-enable of non-reset registers. The chat:
  "BertVT also claimed there was no difference... what? :D". The language doesn't hardcode reset polarity or sync-ness;
  the module that instantiates a design decides.

### 2.5 Registers, wires, variables: four separate words

| Keyword | Meaning |
|---|---|
| `wire` | Module-level signal, driven by exactly one `comb` block, one `clocked` block (via `reg wire`), one instance output, or its declaration expression (`wire w = expr;`). The type and domain can be inferred. |
| `reg` | A register declared **inside** a `clocked` block, for example `reg d0: int(0..256) = 0;`. |
| `reg wire x = 0;` | "Turn this existing wire or output port into a register driven by this block." It works on output ports, wires and even interface fields (`reg wire output.valid = false;`). |
| `val` | Immutable local variable. |
| `var` | Mutable local variable. |
| `const` | Compile-time constant, and also `const { ... }` blocks. |

- **Registers can only be declared inside `clocked` processes** ("registers cannot be declared outside of clocked
  processes").
- **A `clocked` block driving a port without marking it as a register is an error**: "clocked process cannot drive port
  without marking it as a register". You cannot accidentally create a flop.
- Registers can be declared inside **functions**, which gives you the "reg trick" for reusable stateful helpers:

  ```kh
  fn delay(T: type, next: T, reset: T) -> T {
      reg d: T = reset;
      val v = d;
      d = next;
      return v;
  }
  // in a clocked block:
  y3 = delay(int(8), delay(int(8), x, -1), -1);   // two-cycle delay line, one expression
  ```

### 2.6 One assignment operator, software semantics

- **There is only `=`, plus compound forms like `+=`. It "always immediately assigns the value. The following lines in
  the code always immediately observe the newly assigned value. It is the compiler's responsibility to correctly generate
  code that simulates and synthesizes correctly."**
- Registers behave like variables inside their own clocked block. Other blocks see the new value only after the clock
  edge, which is a "double-buffer timing model."
- The Verilog backend does the dirty work. It creates *shadow* local variables for registers (write-through to both the
  signal and the shadow), uses non-blocking assignments for the real register writes, and names every process.
- **No sensitivity lists.** Every combinatorial process is sensitive to everything it reads. The backend even adds a
  dummy signal so no `always @(*)` block ever ends up with an empty sensitivity list, which would mean it never runs, not
  even at startup.

### 2.7 Combinatorial blocks: no latches, no loops

- In a `comb` block, every signal must be driven on **every path** or not at all. A partial assignment fails with **`driver
  mismatch between conditional branches`**, hint: "This would not create valid combinatorial logic. Ensure both branches
  drive the same parts of the signal or do a default full assignment beforehand." **Inferred latches cannot exist.**
- Reading a signal before driving it in the same comb block fails with **`combinatorial self-loop`**: "a combinatorial
  process cannot read a signal before it drives it, this would create a combinatorial loop."
- Driver analysis works at bit, element and field granularity ("cones" and "masks", a "super overengineered system").
  Different processes can drive disjoint slices of an array or different fields of a struct, and the errors name the
  exact parts: `wire w has multiple overlapping drivers`, `port y has multiple drivers`, `is not fully driven`,
  `wire w is not driven` (warning), "parts with multiple drivers", "parts without a driver".
- `dynamic array index assignment to not yet driven signal`: you cannot write `w[i] = ...` with a hardware index unless
  `w` was fully driven first, because that would also infer latches.
- Locals in the generated Verilog are initialized to `'bx` "to avoid inferring latches."

### 2.8 Types

- `bool` (true/false). Bits are just `[N]bool`, and there is no 4-state `logic` type in the language.
- **`int` is an arbitrary-precision mathematical integer, as in Python.** Hardware integers are *range-constrained ints*:
  - `int(0..256)` covers 0 to 255 (exclusive end), `int(0..=buffer_size)` uses an inclusive end, and `int(2..)` means
    "at least 2".
  - `int(8)` with a single integer argument is an 8-bit **signed** range (-128..128). `uint(32)` is 32-bit unsigned.
  - **Multi-ranges with holes** are allowed: `int(..0, 1..)`.
  - Stdlib aliases: `uint = int(0..)`, `positive = int(1..)`, `negative = int(..0)`, `nonzero = int(..0, 1..)`.
    `natural` was **removed** after an embarrassing off-by-one: it originally started at 1, "unlike VHDL." The chat
    response was "why? :/" with a Wikipedia link. A later mix-up with `positive` produced "[Image] not taking any
    chances this time."
- Arrays use a prefix length: `[N]T`, `[num_states][2]bool`. Literals are `[a, b, c]`, repeats are `[false] * N`,
  spreads are `[*buffer[1..], in_data]`, and there are comprehensions: `[half_paths[i][0] for i in 0..num_states / 2]`.
  Zero-width types are supported everywhere.
- **Tuples**: `(a, b)`, `Tuple(T, T)`, and the unit type `Tuple()`.
- **Structs**, generic and with member functions: `struct Pair(T: type) { x: T, y: T }`, constructed with
  `Generator(3).new(generator_0=[...], generator_1=[...])`.
- **Enums are tagged unions, as in Rust/Zig**: `enum ABC { A(bool), B(uint(0..2)), C, D }`. The stdlib provides
  `Option(T)` (`.Some(x)` and `.None`), `Result(T, E)` and `Either(L, R)`. "C and VHDL enums are just the tags, so why not
  add payloads?" Payloads are read only through `match`, so you can never read an invalid payload. Padding bits are zero,
  which keeps every type bit-comparable.
- **Types are first-class values**: `T: type`, `type t_probs = [num_states]uint(prob_bits);`, `type(x)` (typeof), and
  functions that return types. Generic type aliases also exist: `type Pair(T: type) = Tuple(T, T);`.
- Bit conversion members: `T.to_bits(v)`, `T.from_bits(bits)` (only allowed "for types where every bit pattern is
  valid"), `T.from_bits_unsafe(bits)`, and `T.size_bits`.
- References: `ref(x)` and `deref(r)`. "References must stay inside the module they belong to, they cannot escape into
  other items."
- `any` is an escape hatch type that is only checked at elaboration or codegen time.
- Strings are **format strings by default**: `"v_{i + 1}"`. Use `r"..."` for a raw string.
- `undef` is the undefined value, X in Verilog.

### 2.9 Integer ranges, overflow safety, and flow-sensitive implications

- Every arithmetic operation computes a **tight output range**: `uint(0..8) + uint(0..8)` gives `int(0..15)`, and
  subtraction can go negative. Assigning to a narrower type is a **`type mismatch`** error, so implicit truncation can't
  happen. Wrapping must be explicit with `% 2**N`.
- **If-conditions refine types** (flow typing / implications):

  ```kh
  module top ports(p: in async uint(0..8)) {
      comb {
          if (p < 4) {
              var v: uint(0..4) = p;   // OK: inside this branch p has type int(0..4)
          }
      }
  }
  ```

  Without the `if`, the same assignment fails with `type mismatch`. `p == 4` narrows to `4..5`, and `p != 4` narrows to
  the multi-range `0..4, 5..8`. That is how `if (b != 0) { y = a / b; }` typechecks. **Division by zero and modulo by
  zero are compile errors** unless the divisor's range excludes 0. Reassigning the variable erases what was learned.
- `assert(cond, msg)` is a simulation-time and compile-time assert. `assert_assume(cond)` is assert plus "from now on,
  assume this; if it is ever false in silicon, that is undefined behavior." It is how you tell the type system what you
  know, and it lowers to an X on the invalid branch.
- Range arithmetic was checked with Z3 proofs (`python/.../int_range_proofs.py`). From the chat: "I hate it when my
  seemingly trivial compiler feature (computing tight bounds for all integer arithmetic operations) suddenly turns out to
  be a classic crypto problem" (tight modulo bounds).
- The limitation: relational bounds through arithmetic, such as `2 * degree <= iter`, are not derived. Asked whether to
  pull in a full SMT solver, the author called it a "slippery slope." Maarten's joke-but-maybe-not idea is to let AI
  replace `assert_assume` with **Lean proofs**.

### 2.10 Compile-time evaluation (the Zig-like part)

- The whole language runs at compile time: functions, loops, `while`, `match`, recursion, structs, arrays and strings.
  Hardware is what remains after elaboration.
- **`const { ... }` blocks** sit at file level, inside modules, or even inside generic parameter lists. They act as
  in-file unit tests, elaboration checks, and "explicit template instantiation":

  ```kh
  const {
      axi_gearbox(T=bool, input_width=8, output_width=3);
      axi_gearbox(T=bool, input_width=3, output_width=8);
  }
  module foo(a: int, b: int, const { assert(a > b, "Invalid parametrization"); }) ports() {}
  ```

- A **compile-time lookup table** built by a block expression inside a comprehension (from the convolutional decoder):

  ```kh
  const codebook: [num_states][2]bool = [
      {
          val code = uint(length).to_bits(i);
          var output = [false, false];
          for (j in 0..length) {
              output[0] ^= code[j] & generator.generator_0[length - 1 - j];
              output[1] ^= code[j] & generator.generator_1[length - 1 - j];
          }
          output
      }
      for i in 0..num_states
  ];
  ```

- **`ident("name_{i}")` makes identifiers dynamically**, almost anywhere the grammar expects one: wires, ports, interface
  fields, instance connections. It replaces generate-block naming hacks and "swreg-like" codegen tools, and it lets you
  match arbitrary existing Verilog port naming schemes. It is also an escape hatch for keywords: `ident("type")`. The name
  history went `id_from_str` → `@"..."` (rejected) → `ident`.

  ```kh
  for (i in 0..N) {
      pub wire ident("v_{i + 1}"): T[i];
      clocked(clk) {
          reg wire ident("v_{i + 1}") = undef;
          ident("v_{i + 1}") = (F[i])(ident("v_{i}"));
      }
  }
  ```

  This is a generic pipeline whose stage functions are passed in as an array of functions, `F: [N]any`.
- Functions are first-class: `(F[i])(x)`. Functions capture their scope. Generic packages emerge naturally: a function can
  return a configured `struct` type with member constants and functions. Testers built `reedsolo.rs_config` this way, and
  the author's reaction was "That is basically Inigo's generic VHDL packages, but as an emergent capability of language
  features coming together! 😍"
- **Packages are first-class values** (`import lib; fn call(package: any) -> int { return package.f(); }`).
- Loops are unrolled. `break`, `continue` and `return` work even with hardware-dependent conditions
  (`if (i == a0) { break; }` inside a `for` over a compile-time range), and early `return` works from hardware functions.

### 2.11 Interfaces and views

```kh
pub interface axi_stream(T: type) {
    data: T,
    valid: bool,
    ready: bool,

    view input  { data: in,  valid: in,  ready: out }
    view output { data: out, valid: out, ready: in  }
}
```

- An interface is a bundle of signals plus named **views**, which give the direction from one side's perspective. The
  `view` keyword replaced a nested `interface` keyword after a colleague (Jasper) found the double use ugly.
- Ports use `input: interface sync(clk, async rst) axi_stream(T).input`, and internal interface wires use
  `wire w: interface Foo(T);`.
- Interfaces can be generic, and their fields and views can be conditional or looped (`if (X) { x: bool }` inside an
  interface). A tester ("a true believer ❤️") used `if`s inside interfaces before anyone knew it was supported.
- Planned or (per the instructions) done:
  - **interface arrays**, for crossbars, banked memories and N-way dispatchers;
  - an **interface connection statement** with explicit views, so direction is checked and inferred;
  - **private interface fields with public API methods**, so AXI handshakes can only go through a correct-by-construction
    API ("all low-level protocol stuff concentrated in a few classes"), with a greppable `unsafe`-style backdoor;
  - a stdlib `Stream(T: type)` that everyone shares.
- Rule of thumb from the author: "never design interfaces with a bunch (>3) of wires going in the same direction, group
  them in a struct."

### 2.12 Control flow and pattern matching

- `if/else`, `for (i in 0..N)`, `while`, `loop`, `match`, `break`, `continue` and `return`. Parentheses around conditions
  are required: `if (c) { }`.
- `match` supports literals, ranges (`in 2..4 => {...}`), constants, bindings (`val other => {...}`), wildcard `_`, and
  enum variants with payload binding (`.Some(val x) => {...}`). **A hardware `match` must be exhaustive.** Unreachable
  branches produce the warning `unreachable match branch` ("this branch can never match").
- Block expressions: `{ stmts; expr }`.
- "variable has not yet been assigned a value in all preceding branches" is "one of my favorite errors in the compiler"
  (the author).

### 2.13 Imports and file scope

- `import examples.axi_stream.axi_stream;`, `import a.[lib, other];`, `import lib as renamed;`.
- `pub` controls visibility, and `private items cannot be imported`.
- All files are parsed together, with no compile-order dependence. Top-level items can be in any order and may refer to
  each other cyclically; real cycles are reported as `encountered cyclic dependency`.
- A prelude automatically imports the basic std items. The stdlib (`std/types.kh`, `std/math.kh`, `std/util.kh`) is
  embedded in the compiler binary and has "all 29 public items 😂": `min`, `max`, `abs`, `gcd`, `clog`, `clog2`, `print`,
  `assert`, `assert_assume`, `bool_to_int`, `Option`, `Result`, `Either`, ...
- Import errors suggest fixes sorted by edit distance ("did you mean").

### 2.14 Naming conventions

Types, modules, interfaces and views use UpperCamelCase. Functions, variables and signals use lower_snake_case. The demo
code itself is often all lowercase.

---

## 3. What the compiler does

### 3.1 Architecture (Rust workspace `rust/`)

- `khdl_language` is the core library:
  - `syntax`: hand-written tokenizer, LALRPOP grammar with parser error recovery, formatter, and manifest parsing.
  - `front`: elaboration, expression evaluation, the type system, range arithmetic, implications, domain checking,
    flow/SSA-style variable merging, match coverage, diagnostics.
  - `mid`: IR, IR validation, cleanup, **cones/masks** (driver and combinatorial-loop analysis).
  - `back`: `lower_verilog`, `lower_cpp`, `lower_verilator` and `wrap_verilator`.
- `khdl_bin` is the CLI: `khdl build --manifest khdl.toml [--top ...] [--output-verilog ...] [--output-cpp ...] [-j N]`
  and `khdl fmt [--check]`.
- `khdl_lsp_server` is the LSP: diagnostics, go-to-definition, find references, semantic tokens, folding, selection
  ranges, and formatting.
- `khdl_python` provides the Python bindings via PyO3, published as `khdl` on PyPI.
- `khdl_wasm` powers the in-browser playground at khdl-lang.org. It has an IR output tab and dark mode, and playground
  links encode the source in the URL (`?source=cb64-...`).
- `khdl_std` is the standard library and `khdl_util` holds shared utilities.
- `lsp_client/` is a VS Code extension with a TextMate grammar (`khdl.tmLanguage.json`), fuzzed against the real
  tokenizer.
- `fuzz/` has fuzz targets for the parser, the formatter (idempotency and token preservation) and the TextMate grammar.
  There is also a Python-driven random-expression fuzzer that compares compile-time evaluation against Verilator
  simulation.

### 3.2 Checks performed on every compile

- Strong type checking of every assignment, connection, argument, return and condition.
- Integer range inference and overflow/truncation detection (`type mismatch`), division and modulo by zero, array index
  and slice bounds (`array index out of bounds`), and negative widths.
- **Clock/reset domain crossing checks** (`invalid domain crossing: async to sync / different clock / different reset /
  non-clock to clock / clock to sync`), including through module instance ports and control-flow conditions.
- Register discipline: registers only in clocked blocks, reset values present or `undef`, no register driving an input
  port, no clocked block driving a port that isn't `reg wire`.
- Driver analysis: multiple drivers, overlapping drivers, undriven and partially driven signals, and drivers that differ
  between branches (no latches).
- Combinatorial loop detection (`combinatorial self-loop`) and cyclic module instantiation.
- Use-before-assignment of variables, exhaustive hardware `match`, and unreachable branches.
- Assigning to input ports (`cannot assign to input port`) and to `val`s (`cannot assign to immutable variable`).
- Interface view and port direction mismatches.
- Hardware versus compile-time context errors ("signal evaluation is only allowed in a hardware context").
- `assert` failures during elaboration, with stack traces through the elaboration hierarchy.

### 3.3 Backends and simulation

- **Verilog 2005 output**: a simple, lowest-common-denominator subset that is deliberately paranoid about Verilog's
  "suspicious bitwidth inference rules". It casts with `$signed`/`$unsigned`, widens via `0+`, and truncates through
  temporaries of known width. Loops are fully unrolled. A cleanup pass strips SSA temporaries such as
  `z/z_0/z_1/z_2`. The author on the early output: "I had already asked for a summary of it myself so I could facepalm a
  bit."
- **C++ backend**: generates C++ that simulates the design directly without any EDA simulator. It uses a double-buffered
  prev/next per signal and one function per process.
- **Verilator integration**: `module.as_verilated(build_dir)` → `.instance(trace_path=...)` → `ports.x.value = 1` →
  `sim.step(1)`. There are options for Verilator optimization level, C++ compiler choice, and ccache (`OBJCACHE=ccache`).
  The ".so is randomly suffixed so `dlopen` doesn't hand you a stale cached binary" fix came from feedback.
- **Planned built-in simulator** (on the `sim` branch): every process is compiled as an **LLVM function**, which is
  highly parallel and ccache-style cacheable. The simulator is **delta-cycle-free** and schedules process order
  statically. The perf ideas include clock-gating-style idle skipping, multithreading and partitioning. Motivation:
  "commercial tools are a disaster for agentic and RL setups"; fast compile plus simulate makes KHDL ideal for AI-driven
  RTL generation.
- **Python as the testbench language.** "Testbench setups = 4 lines of Python 😛." You `resolve` any item, call KHDL
  functions directly from Python with automatic value conversion, elaborate generic modules with Python kwargs, and
  sweep generics in pytest. Example:
  `axi_gearbox = compiled.resolve("examples.axi_gearbox.axi_gearbox"); top = axi_gearbox(T=bool, input_width=7, output_width=9)`.
  "Pytest runs 380 tests in 4s, and more than half of them build a Verilator testbench."
- Further output ideas from the design doc: C headers for register maps and structs, VHDL output, a Lean/SMT backend for
  formal work, and parsing external Verilog/VHDL to infer domains automatically ("be so good at this that people
  consider using this compiler just for its Verilog/VHDL features, a bit like Zig").

### 3.4 Tooling quality and performance

- **Modern error messages** use the `annotate-snippets` style familiar from Rust: primary span, info spans, `= info:`
  footers and `= help:` hints. Each test asserts exactly one diagnostic, there is a dedicated "check against internal
  compiler errors" effort, and the git log contains many commits like "Improve wording of a bunch of diagnostics."
- The **auto-formatter** (`khdl fmt`) is idempotent and fuzzed. `// fmt: off` regions keep lookup tables from being
  exploded "one element per line".
- The compiler is **fully multithreaded, down to individual file-level items and elaborated modules**. Its query-based
  architecture guarantees it "never deadlocks, even if you're doing cursed things (e.g. `struct Foo(T: type) { x: Foo(T) }`)".
  There is a generated-project benchmark for compile speed, profiling via samply/valgrind, and a string interner. "Unlike
  VHDL 😠", elaboration and type checking are fast. One lesson: release mode was 8x faster than the accidental debug
  build.
- The LSP is built into the compiler, so editor behaviour equals compiler behaviour. Generic bodies are checked through
  `const { }` instantiations. Diagnostics from stdlib asserts are reported at the user's call site.

---

## 4. Concrete frustrations with existing HDLs that KHDL fixes

| Pain (SV / VHDL / Chisel) | KHDL answer |
|---|---|
| **Implicit truncation/extension.** SV: "values of any type are assignable to any other type, the bit widths don't even have to match." Width-context rules make `a + b` silently lose its carry. | Range-typed ints with exact result ranges; narrowing is a `type mismatch`; wrap only with an explicit `%`. |
| **Overflow unchecked** without a separate linter. VHDL constrained integers are checked only at sim/synthesis time. | Overflow is a *compile-time* error. `assert_assume` turns it into a runtime check only where you choose. |
| **Manual bit-width arithmetic** (`$clog2(N+1)-1:0`, off-by-one widths). | `int(0..=N)`: write the values, not the bits. |
| **Blocking vs non-blocking** (`=` vs `<=`), VHDL signal vs variable, delta-cycle reasoning: "a historic artifact of the early event-driven models of the languages, but there is no need to repeat this design mistake." | One `=`, immediately visible; the compiler generates correct NBA/shadow code. "No delta cycles!" |
| **Sensitivity lists**: incomplete `always @(a or b)` and sim/synth mismatch. | None exist. `comb {}` is sensitive to everything; there is never an empty list. |
| **Inferred latches** from a missing `else` or default. | `driver mismatch between conditional branches` makes it a compile error. |
| **Accidental registers / reg-vs-wire confusion.** SV `logic` is a register, wire or temp depending on context; VHDL `signal` and `variable` are each two things. | Separate `wire`, `reg`, `val`, `var`, `const`; clocked blocks must explicitly `reg wire` a port. |
| **Forgotten reset / random power-up state.** | A register needs a reset value or an explicit `undef`. |
| **Reset polarity and sync-vs-async baked into every module.** | Resets are part of the domain (`sync(clk, async rst)`), the instantiating module decides, and the reset style is a backend choice. |
| **Clock domain crossings** found late by a separate CDC tool ("left to a separate linting or domain crossing checking step late in the development pipeline"). | Domains are part of every type and checked on every compile, with a greppable `unsafe_value_with_domain`. |
| **X-propagation pessimism/optimism; "stdlib functions that warn on X are just wrong. X should propagate!"** | `undef` is first-class; X is assigned to non-reset regs during reset; `assert_assume` → X; padding is defined. |
| **Generate blocks & genvar naming hell; Python/Perl/Jinja codegen; editor templates for instantiations.** | Plain `for`/`if` in any list (ports, interfaces, connections, statements) plus `ident("...")` and a real compile-time language. |
| **Parameter hell**: parameters can't be types or functions; VHDL generics only recently got types; C++-template-style errors leak out of instantiations. | Generics are values, types, functions, modules or packages. Ports can appear or disappear based on generics. Items typecheck on their own. |
| **SV interfaces/modports with poor vendor support; huge port lists duplicated everywhere.** | `interface` + `view`, generic interfaces, `ports(clk, rst, x)` shorthand, grouped domain blocks. |
| **No import system in SV** ("the user is responsible for compiling all files in the right order, and names always resolve to the last definition with a matching name"); VHDL libraries plus compile order. | A real package hierarchy with order-independent, all-files-at-once parsing; packages are values. |
| **Variables declared only at the top of a process/architecture.** | Declare anywhere, with type inference. |
| **Operator precedence traps** (`a & b == c` in Verilog/C). | No mixing of `& | ^` or `&& || ^^` without parentheses; comparisons are non-associative. |
| **VHDL type system "too strong in some ways"** (same-shape arrays not assignable) and "too weak in others" (ranges unchecked, types checked only at run or synthesis time). | Structural array types; everything is checked at elaboration. |
| **Cryptic EDA error messages** ("EDA tools set the bar very low"). | Rust-style diagnostics with spans, info and hints. |
| **Tool support lag**: SV-2017 features unsupported by vendor X for years. | Output is Verilog 2005; the language evolves in the compiler only. |
| **UVM boilerplate** (`uvm_info(get_type_name(), $sformatf("..."), UVM_LOW)`). | Python/pytest testbenches with format strings and `print`. |
| **Chisel**: Scala's `if`/`else` are claimed for elaboration, so hardware control flow is `when() {}.otherwise {}` callbacks; Scala syntax. | Real `if`, `match`, `for`, `while` and `return` work for both compile-time and hardware values in one language. |
| **Slow commercial tool turnaround** for iterate/agentic loops. | Multithreaded compiler, cached Verilator, and an LLVM-based simulator. |
| **Lint as an afterthought** (Spyglass, Verilator lint, code review). | "Compiler check" instead of "manual code review and linting, both of which have much more latency." |

---

## 5. Roadmap and future plans

Per the instructions, everything below can be treated as done. It is listed so lyrics can draw on it.

- **Built-in simulator.** LLVM-compiled processes, delta-cycle-free, statically scheduled, with idle/clock-gating
  skipping, multithreading and partitioning, and ccache-like caching. It targets fast agentic and RL hardware loops.
- **Reset strategy options** in the backend, and X assignment to `undef` registers during reset.
- **Interface arrays**, an **interface connection statement** with checked views, **private interface fields with API
  methods**, interface reflection, and Python access to interfaces.
- **Query-based compiler rework.** No deadlocks on cursed recursive types, independent resolution after an error, real
  **stack traces** for elaboration errors, and deterministic diagnostic ordering.
- **LSP**: hover types, autocomplete, rename, auto-import, find usages across files, "show generated RTL for this
  module", diagnostics for uninstantiated generic bodies, and the LSP bundled into the main binary.
- **Formatter**: `// fmt: off` / `on` / `skip` and compact array layout.
- **Verilog output cleanup**: fewer temporaries, no `0+` noise, folded constants, a generalized IR repeat operator, and
  loop-iteration comments.
- **Language conveniences**: `.Variant` shorthand enum construction, `undef(T)`, `reversed()`/`enumerate()`/`zip()`,
  affine range implications (`2*a < 7`), relational bounds, `Fraction` and fixed-point types, one-hot enum register
  annotations, sequential/state-machine-as-code processes (an `@` wait-a-cycle syntax was sketched), the "struct members
  order-independent" fix, and possibly a "clippy for RTL."
- **Interop**: parse external Verilog/VHDL to infer domains, VHDL output, C headers for register maps, a Lean/SMT formal
  backend, and possibly CIRCT.
- **Docs**: an mdbook, splitting the website into index, book and playground, a PyPI release (done), and getting-started
  guides.
- **Process**: bulk-port open-source cores (CVA6, Ibex, Ethernet and AXI projects already have Python test harnesses
  sketched in `design/`) using AI agents, collect every friction point, and tune an "LLM skill" for KHDL.

---

## 6. Notable feedback themes (from `ignored/feedback`)

The main feedback source is Maarten Baert, a colleague and friend, who had an AI agent port his VHDL **Reed–Solomon
RS(240,224) encoder and five-stage decoder** to KHDL. The AI agents involved were "Sol"/"Astra"/"Luna" (OpenAI) and
Opus 5.5. The conversations are in Dutch.

1. **Missing bitwise operators were the #1 pain.** Before the fix, `a ^ b` on `uint(8)` failed with "operator requires
   type bool", and `flag == true` failed. The workaround (`to_bits` → per-bit xor → `from_bits`) bloated the Verilog to
   **3.8 MB for the encoder and 7.6 MB for the decoder** ("absurd"), multiplied by full loop unrolling. Bitwise operators
   and equality on everything were added.
2. **Range analysis is strict but sometimes needs help.** `2 * degree <= iter` doesn't imply `iter + 1 - degree >= 0`.
   Agents reached for `assert_assume`, and the VHDL-ism `bounded()` helper was removed after 48 calls. Padded unrolled
   counters raised the question of whether KHDL's exact ranges cost an extra bit.
3. **Domain errors are correct but surprising for synchronous-reset habits.** Reading an `async` reset inside
   `clocked(clk)` gives "async to sync"; the answer is `clocked(clk, sync rst)`. People want a documented sync-reset
   idiom so AI agents don't reach for `unsafe_value_with_domain` "merely to silence the diagnostic."
4. **"Generic modules are functions" surprised users.** A non-generic module can't be called with `()`, and adding
   defaulted generics forces call sites to change. Maarten compared it to "the same situation as Python decorators."
5. **Tooling papercuts**: stale Verilator `.o`/`.so` reuse (the `dlopen` cache: "do I really have to rewrite dlopen
   myself? That's so cursed."), a hardcoded `g++`, `--trace` always on, Python 3.14 metadata, debug-mode builds running
   8x slower, and an LSP that didn't work on an ancient VS Code 1.75.
6. **Compiler bugs found by the AI**: a cross-file `Span::join` panic on indexed assignment to an imported interface
   field ("Chien triggers a compiler panic"), `clog(1, n)` never terminating, and LSP silently dropping stdlib assertion
   failures.
7. **Formatter vs. lookup tables**: 256-entry tables exploded to one element per line, leading to `// fmt: off`.
8. **Naming bikesheds**: `id_from_str` → `@"..."` → `ident`, `interface` inside `interface` → `view`, `ceil_div` → `cdiv`
   → **`+/`**, and `natural` was deleted.
9. **Delight moments**:
   - the generic `rs_config` "is really Zig-like, exactly how I hoped it would all come together";
   - "Pretraining is overrated" (the AI learned KHDL from the repo alone);
   - "I really want to use KH to generate RTL with AI, partly because then we're rid of all those ridiculously bad tools";
   - "Just a fully 'batteries included' modern language";
   - driving disjoint struct fields from different processes ("Huh, I didn't know this was possible");
   - "Once there's really nice and convenient test infrastructure, writing tests is actually fun."
10. **Design debates**: private interface fields versus Rust-style module privacy ("Is this a trojan horse to force me to
    make public the default? 😅"), Chisel's popularity in academia (the "slow clash" of legacy SV/VHDL vs Chisel; "Neither,
    AI will write everything ;)"), and whether loops must stay loops in Verilog.

---

## 7. Lyric-worthy nuggets (more than 40)

### Keywords and syntax hooks

1. **`comb` and `clocked`**: the only two kinds of process. "Two kinds of blocks and no third way."
2. **`reg wire`**: "put a flop on that wire." The keyword pair says "this wire is now a register."
3. **`= undef`**: the only way to skip a reset is to *say so*. "I didn't forget the reset, I declared it undef."
4. **`sync(clk, async rst)`**: domain as a type annotation. "My data knows its clock."
5. **`unsafe_value_with_domain`**: the greppable sin, the only door across a clock boundary. "Every crossing signs its
   name."
6. **`unsafe_bool_to_clock`**: making a clock out of a bool is something you admit to.
7. **`int(0..256)`**: "I don't count bits, I count values." Also `int(0..=N)` and `int(2..)`.
8. **`uint(32)` with `% 2**32`**: wrapping is an explicit choice, never an accident.
9. **`+/`**: the ceil-division operator, for every `(a + b - 1) / b` you have ever written.
10. **`^^`**: boolean XOR gets its own operator.
11. **`ident("v_{i}")`**: names conjured from strings, the end of genvar naming hell.
12. **`const { }`**: compile-time blocks double as unit tests and explicit instantiation.
13. **`view input` / `view output`**: one bus, two points of view, no modport mess.
14. **`instance child ports(clk, rst, x, y);`**: when the names match, you don't repeat them. No more `.clk(clk)` x 200.
15. **`external module`**: wrap the legacy IP and carry on.
16. **`pub`, `import a.[lib, other];`**: a real module system for hardware.
17. **`ref` / `deref`**: pointers that "cannot escape" their module.
18. **`Option(T)` with `.Some(val x)`**: valid/data as a type. "Never read a payload that isn't there."
19. **`assert_assume`**: "assert it, then assume it, and if silicon disagrees, that's UB."
20. **`N: int(2..)`** on a synchronizer: a one-flop synchronizer will not compile.

### Error messages (exact strings)

21. **`invalid domain crossing: async to sync`**
22. **`invalid domain crossing: different clock`** / **`different reset`** / **`different clock and reset`**
23. **`non-clock to clock`** / **`clock to sync`**
24. **`help: to intentionally cross domains, use `unsafe_value_with_domain` or `unsafe_bool_to_clock``**
25. **`driver mismatch between conditional branches`**, with the hint "This would not create valid combinatorial logic."
    This is the latch killer.
26. **`combinatorial self-loop`**: "a combinatorial process cannot read a signal before it drives it."
27. **`clocked process cannot drive port without marking it as a register`**
28. **`clocked block without reset cannot drive register with reset value`**, with the hint "either add a reset to the
    block or use `undef` as the reset value."
29. **`registers cannot be declared outside of clocked processes`**
30. **`wire w has multiple overlapping drivers`** / **`port y has multiple drivers`** / **`wire w is not driven`**
31. **`variable has not yet been assigned a value in all preceding branches`** (the author's favorite error)
32. **`division by zero is not allowed`** / **`modulo by zero is not allowed`**, which are compile errors decided by
    range types.
33. **`type mismatch`**: the error that replaces silent truncation.
34. **`hardware match statement must be exhaustive`**
35. **`cyclic module instantiation`** / **`encountered cyclic dependency`**
36. **`cannot assign to input port`**
37. **`array index out of bounds`**, decided by types at compile time.
38. **`from_bits is only allowed for types where every bit pattern is valid`** (use `from_bits_unsafe` if you dare)

### Old way vs KHDL way

39. `always @(a or b)` (forgot `c`) → `comb { }` reads everything and never misses one.
40. `<=` vs `=`: "which one goes in the clocked block again?" → one `=`, and "the following lines always immediately
    observe the newly assigned value."
41. Missing `else` creates a latch → a compile error with a hint.
42. `assign sum = a + b;` silently drops the carry → `int(0..15)` is computed, and narrowing it won't compile.
43. A CDC checker run at tapeout minus three months → a CDC check on *every* compile.
44. `generate for (genvar i...) begin : gen_blk` → plain `for` plus `ident("x_{i}")`.
45. `$clog2(N+1)-1:0` → `int(0..=N)`.
46. A Python script that generates Verilog → the compile-time interpreter *is* the script.
47. `uvm_info(get_type_name(), $sformatf("..."), UVM_LOW)` → four lines of pytest.
48. Chisel's `when().otherwise()` → just `if`/`else`.
49. A "logic" that might be a wire, a flop or a latch → `wire`, `reg`, `val`, `var`.
50. "Compile all files in the right order or the last definition wins" → order-independent packages.
51. Delta cycles → "No delta cycles!"
52. Reset polarity hardcoded in 400 modules → the reset is part of the domain, and the parent decides.

### In-jokes and story beats

53. **The `natural` that started at one**: "why? :/". It was removed, and then `positive` briefly included zero:
    "[Image] not taking any chances this time."
54. **"PTSD with existing RTL languages"**: the author's diagnosis of defensive HDL coding habits.
55. **The 7.6 MB decoder.v**: what happens without bitwise operators and with every loop unrolled.
56. **The dlopen ghost**: delete the `.so`, rebuild it, and Linux hands you the old one. "Do I really have to rewrite
    dlopen myself? That's so cursed."
57. **"Kotlin for hardware"**: interop-first second-generation language.
58. **"Decouple language development from tooling development"**: stop waiting on the EDA vendors.
59. **Cursed types**: `struct Foo(T: type) { x: Foo(T) }` must not deadlock the compiler.
60. **"Chien triggers a compiler panic"**: Reed–Solomon versus the span joiner.
61. **"Pretraining is overrated"**: an AI learned KHDL from the repo alone.
62. **Tight bounds for modulo turned out to be "a classic crypto problem."**
63. **Lean proofs instead of `assert_assume`**: formal methods as a punchline that might come true.
64. **".kh is in the training data now"**: the name chose itself.
65. **BertVT's "there's no difference"** (between reset styles): there was, in the clock enable.
66. **The Verilog backend's paranoia**: `$signed`, `$unsigned`, `0+` and temporaries everywhere, because Verilog's
    width rules can't be trusted.
67. **"Batteries included"**: compiler, formatter, LSP, Python, simulator and playground in one binary or ecosystem.
68. **"Testbench = 4 lines of Python."**
69. **"380 tests in 4 seconds."**
70. **"Generic modules are just functions that return modules"**, which is elegant and confused everyone at least once.
71. **"With great power comes great responsibility!"**: the author's reply when a user started pushing the generic
    programming hard.
