## Design guidelines

* Module/Functions/Structures/... should typecheck on their own, independent of their users.
    * This means that an error is _either_ inside of something or at the use site. It can't propagate outwards.
    * Unlike C++ template parameters!
* Prefer stdlib over language features
* Privilege the stdlib as little as possible, libraries should be able to do the same
* Put enough semantics into the language to avoid any possible backend errors.
    * Examples: clocks should be marked as such, values should know which clock if any they are synced too.
    * No delta cycles!
* Interoperate with VHDL, Verilog and SystemVerilog as much as possible
* Don't raise the abstraction level too much. The goal is not to invent something completely new, just to drastically improve the ergonomics.

## Sources of inspiration

* Software:
  * Python:
    * convenient to use and start new projects in
    * very convenient `int` type
  * Rust: 
    * safety: preventing as many bugs as possible by making the compiler check for them continuously during development
    * very strong type system
    * cargo: very easy to start a project, compile, publish, version, declare dependencies 
  * Kotlin
    * a "second generation" language that replaces Java while still having very good interop with existing Java code
* Legacy RTL languages
  * (System)Verilog
  * VHDL
* New RTL languages (see also https://github.com/drom/awesome-hdl) 
  * MyHDL
  * Chisel

## Specifics

### Syntax

* Semicolon, allows spreading expressions over multiple lines.
* `{}`
* `()` for params and generics (which are really just params!)
* `[]` for array indexing and literals
* `<>` only for comparison operators, no generics! (simplifies parsing)
* expression based wherever possible (eg. loops, if, ... are all expressions)

Alternative: use `[]` for generics and `()` for array indexing
  but then what about array literals?

### Identifiers

Standard stuff, maybe borrow r# prefix for keywords from rust?

Expose name mangling as a library?

### Types

Prefer types defined in the stdlib over built-in types.

Standard:

```
Bool
Int, UInt, Int(min=..., max=...)
B(n), I(n), U(n) -> B8, I8, U8
IeeeFloat(signed, mantissa, exponent) -> F32, F64
FixedPoint(signed, integral, fractional) -> UF[4, 3], IF[4, 3]

List(T)
Array(T, n)
```

Integers are infinite bits by default (like python). The rest of their design follows https://www.jandecaluwe.com/hdldesign/counting.html.

What to do about arrays? See https://old.reddit.com/r/ProgrammingLanguages/comments/vxxfh2/array_type_annotation_syntax_string_vs_string/ for some great discussion. Options/summary:

* Rust-like `[T; N]` or `Array(T, N)`
    * easy to parse, clearly different from array literals
    * works badly for multi arrays, eg. `[[T; N]; M]` is indexed as `t[m][n]`
* C-like `T[N]` is easy to type
    * corresponds nicely to indexing order
    * might be ambiguous with indexing into arrays/lists of types? is that a problem?
    * tricky to parse? not really
* Post suggests `[N]T`, but that's really weird and doesn't match usage
    * but maybe trying to match usage was the whole mistake of the C type notation?
* What about allowing multiple dims? also for expressions!
    * `[T; N, M]`, `Array(T, N, M)`, `T[N, M]`
        * for now let's go with this idea combined with `Array` the most

User-defined types:

```
enum Letter { A, B, C }
struct Point(type T) {
  x: T,
  y: T,
} 
```

Do we want types to be first-class? Probably best, so we can just pass them as ordinary arguments. This means that type constructors should also just be functions.

Allow type and function declarations everywhere.

Utility types:
* `Option(T)` like Rust
* `TriState(T)`: should it _actually_ be tristate, or just have a  

### Parameters

Parameters in this language are a unification of generics and normal parameters in other languages. They're used for both types and functions. (and modules if we decide to add them)

Goals:

* types and values can be freely mixed
* parameters and types are _immediately_ in-scope for the following parameters
* allow anonymous parameters in type declaration and maybe even in functions
* allow conditional parameters

Example:
`(A: type, b: A, c: uint, d: Array(A, c))`
`(enable: bool, if enable { c: int }, d: u8`

### Functions

Distinction between functions and procedures:
* functions
  * only compute values
  * are combinatorial/compile time
  * specific `pure` class of function that _only_ depends on args and no rng state, outside io, ...?
* procedures
  * can use time, clock, registers, ...
  * have inout/out parameters!
* Both 
  * can return values!
  * input arguments


### Testing

Do we want (only) our own simulator or rely on existing ones? We need interop with existing ones anyway!

#### constrained randomization

* allow constraining multiple values at once
* always explicitly list randomized fields, not with separate setting per field like in SV
* only allow this for simple structs, not for full classes (will the language even have classes?)
* don't bother with default constraints in structs, they're just weird
  tests:

#### Testbench
* no sequencer/monitor/port... semi-built-in stuff, just simple classes that communicate over channels
* no stage stuff, no separate build and connect
* no global shared state through a dict!

* for tests we definitely need non-clocked stuff, so we might as well add it to the language

#### Utilities

* printing debug strings with formatting needs to be super easy
  * (we don't want to repeatedly type `uvm_info(get_type_name(), $sformatf("..."), UVM_LOW)`)
  * different log levels (error, warning, info)
  * different verbosity for info? with some concrete guide on how those levels should be used

* Coverage built-in as much as possible

### Macros

Hopefully functions and the rest of the language are designed well enough that we don't need them.

Maybe have some way of inserting literal VHDL/SV code as an escape hatch? Similar to how C/CPP allow inline assembly.
Maybe even with the same IO management, although instead of register allocation it will do name mangling here.

### Interop

#### RTL

The top-level compilation generates standard VHDL and verilog.
Modules that only use supported features can also be converted to VHDL and Verilog.

We can import and use _all_ existing VHDL and (System)Verilog modules.
* We'll need a VHDL/SystemVerilog parser and type mapping for this.
* What about differing `Z`, `X`, ... support? Eg. do we always assume their values are tristate?

#### Simulators

#### Embedded SW

The compiler should be able to generate C headers for all types (enums, structs, ...) so it's super easy to do memory maps and configuration registers.

#### Programming

Ideally we'd have an easy python/C++ interface for both the compiler itself and and runtime.

Compiler sketch:
```python
project = Project()
project.add_module_path("name", "path")
project.add_module_source("name", "src")
project.compile_top()
```

also expose incremental stuff for easy interop with other build systems. Ideally even `Make` just works decently out of the box.

Runtime simulation sketch:
```python
sim = Simulation()
sim.module["my_module"].send_inputs("b", np.array([1, 2, 3]))
result = sim.module["my_module"].get_outputs("a")
assert result == [...]
```

The rough idea is then that drivers and monitors are implemented in RTL, while python/C++/Rust is used for stimulus generation and output checking. Ideally drivers are written in such a way that the outside language is not _in_ the loop, since that can slow things down. It's still _possible_ to do so for maximum flexibility. Also none of this driver/monitor stuff is hardcoded in the language, it's just a semi-std library.

### Undefined/Impedance

Do we want `Z`, `X`, `H`, `L`? For all values or only special types? Eg. `TriState[T]`.

Probably undefined for everything to detect startup bugs.

### Interfaces

* Interfaces need to be super easy so we can avoid any tooling that needs to generate entity instantiations.
* We want at least bidirectional, do we need more?
* Give users the freedom to pick their own names (slave/master, parent/child, controller/controlled, ...)?
* What syntax do we use for port directions that switch? Or does everything switch?
* Can interfaces have built-in asserts/drivers/monitors? How do we turn them on or off?

* Every `def`/`function`/`module`/... defines an implicit interface.
  * We need an easy way to partially compose them.

### Port directions

Is `in`, `out`, `inout` fine? Do we want more? Or no `inout`? Do we want a separate type for tristate? Or will a handle type fully handle that.

Maybe have `inteface` as a separate port "direction"? That nicely maps to the direction being _inside_ the interface!
Convention: the name used means that _we_ are that name. Eg. a slave could have this signature:
`my_axi_slave(clk, axi: interface Axi4Lite.Slave)`

Do we want `clk` and `reset` to be types or port "direction"s?


### Garbage collection

First thoughts:
* Everything (appears to be) garbage collected and object-oriented? (like python)
* Lists etc don't use pass my copy, also just like python.
* What about mutability? Eg. functions writing to their parameters? Do we just ban this entirely? We kind of want this for unit testing at least.
    * Maybe classes exist and behave differently?
    * Or some specific type for shared stuff, eg. `&` or `Ptr(T)`?
* Do we use Rust reference semantics but with garbage collecting?

### Assignments

Clock/Reset/State:
* Do we want separate clocked vs non-clocked assignments?
* How do assignments know which clock to use? Infer from LHS and RHS (which must match)? Or have some default clock per function?
  * If there are multiple clocks, explicitly set the current clock within some scope?
* How to deal with reset values for delays?

Typing:
* allow automatic widening of types? or with a separate operator?
  * literals automatically widen anyway, so this would only be for arithmetic expressions 
* maybe stick to `truncate`, `extend`, ... for consistency at first

```
a = b
a <= b
a := b
a <- b
a <== b
a <-- b

a = b

reset(!resetn) {
  a = 0
}
delay(clk) {
  a = b
}
```

### Sequential flow

For simple protocols and IO stuff, or other simple state machines, we need some sequential programming model. Again, how to handle resets?

Do we want full fork/join support like in SystemVerilog tasks? Is it easy to efficiently compile this stuff?

```
// simple 10x clock divider
sequential(clk) {
  loop {
    clk_out = 0;
    wait(5)
    clk_out = 1;
    wait(5)
  }
}

// pulse wave generator
sequential(clk) {
  loop {
    wave_out = 0;
    while !trigger_in { wait(1) }
    wave_out = 1;
    wait(1);
    wave_out = 0;
    wait(1);
    wave_out = 1;
    wait(1);
  }
}
```

Do we also want a similar show for pipelined combinatorial stuff? How should the registers be spread out:
* automatically by our compiler?
* automatically by backend -> this doesn't seem to work great in practice
* manually by the user
  * do we want some warnings/errors if values get misaligned?

### Data structures

list: backed by a dequeue

* remove(index), pop/push both sides (back is implicit), len, ...
  array: simple fixed len array
  ranges: self-explanatory
  option: unwrap

### Literals

* decimal, hex, bin, ...
* enum literals: `.A` prefix infers enum type!
* `_` separator? or `'`? (then we reserve _ for wildcard)
* strings, chars (with escape sequences)
* f-strings for easy formatting
* `*` for wildcards? no!

```
true, false
0, 1, 0xA
```

### Expressions

Scalar:

* basic arithmetic, boolean operators, bitwise operators, ...
* array indexing and slicing
* steal `<=` chaining from python? or just rely on `&&` for now?
  * maybe just have an `in` operator for ranges?

Reshaping:

* concatenation operator? bitflip operator?
* convert bits to array and back? transpose arrays? ...

* Is a spread operator like python enough?
  * No, we want easy support for both little and bit endian!
* Are a couple of conversion functions enough?
* Check what VHDL and SystemVerilog do.

### Control flow

* if, while, for, loop, match
* match expressions with bit/hex patterns, don't cares, ...

### Clock interactions

### Clock specifier

How to specify the clock values are synchronized too?

Should the clocking be part of the type or of the specific values? Maybe have clocks be implicit by default, but with some option to specify them?

`def flip_flop(clk: Clock, x: Bool[clk])`
or
`def flip_flop(clk: Clock, x[clk]: Bool)`

Either way, if there's only one clock we'll assume that one by default:
`def flip_flop(clk: Clock, x: Bool)`

### Type conversion

We don't want _any_ implicit type conversion at all.

Most types should have a `to_bits` function for serialization and for putting things on busses.

Between all integers there should be different casts:

* `cast_expand`: only (compile-time) allowed if every value will fit
* `cast_assert`: only (run-time) allowed if the current value fits
* `cast_wrap`: wrap overflowing values around
* `cast_trunc`: truncate overflowing values

### Modules vs entities vs blocks vs functions

Things we need:

* pure compile-time "functions"
* runtime "functions" that can contain delay, make registers, clock, synchronization, ...

### Coding conventions

Capitalization:

* Types: upper camel
* Constants: upper snake
    * What about generics? They're going to look a lot like parameters!
* Everything else: lower camel or lower snake?

### Module system

* Allow cycles!
* Mostly implicit from file system with option to create additional nested namespaces in a file?
* private/public functions and modules

### Compiler flow

Big picture:
* parse
* type/semantics-check each module _independently_, converting to some HIR
  * correctness/typechecking of a module using another module can only depend on the interface on that module, not the code inside
    * except for compile-time asserts?
      * or do we force compile time asserts _into_ the interface itself?
    * this is also important for incrementalness, users only need to wait on parsing the interface and functions used in there
    * language design: support this stuff 
      * making counting matching {} easy?
* walk top-down, instantiating each module with the right parameters and converting to LIR
* convert LIR to output, either Verilog or VHDL

Details:
* lazy, incremental, ...
  * use salsa? https://rustc-dev-guide.rust-lang.org/salsa.html
* for ideas on what we need to support IDEs: https://github.com/rust-lang/rust-analyzer/blob/master/docs/dev/architecture.md 

Worklist: 
* Items can heavily depend on each other, on the result of typedefs, on symbols imported ...
* => We need some kind of worklist algorithm that's very flexible.
  * Work on an item until we get stuck, then move to the next one.
  * If all items are stuck there's a cyclic dependency in the source code, report this as an error.
  * Is `async` rust a good fit here?

### Use cases to examine

* basic pipelined datapath (with backpressure)
* IO state machine
* AXI config register banks
  * is the language powerful enough that we don't need to generate code?
* repeating grids of components
* FIFOs
* CPU instruction decoding
* clock domain crossings, synchronizers, ...

### Build system

### Testing

* easy within-language testing
* easy connection to python and C++ for test vectors
  * they don't need to be timing aware, just passing data back and forth is enough
* easy json parsing on the language side
* regex matching? maybe even at runtime codegen if the regex is simple enough?

### IDE plugins

Utilities:
* invert if
* switch between if/else chain and match
* show type of expression or value of constant (if possible)

### Package manager

### Safety features

We want to detect as many issues as early as possible, primarily during compile time and if not at least with asserts during testing.

Examples:
* we have a very strong type system, and encourage the user to define more types whenever possible
  * add a way for users to constrain their types, with constructors that assert or return `Option`?
* integer range issues are caught by range checking (part of int types), and the user has to explicitly truncate or wrap
* clock/async issues are caught by all values/types having an associated clock property!
  * this forces users to insert synchronization primitives
  * users can opt-out via some `unsafe`-style block
* can we fully detect race conditions between different `async` and `sync` blocks at compile time?
  * investigate this
  * we want to avoid any "unpredictable" or undefined behavior, we want least want to emit "X" if this happens in simulation
* only allow non-resetting registers in pipelines or similar structures that are dataflow-only
