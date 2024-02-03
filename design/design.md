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

## Specifics

### Syntax

* Semicolon, allows spreading expressions over multiple lines.
* `{}`
* `()` for params and generics (which are really just params!)
* `[]` for array indexing and literals
* `<>` only for comparison operators, no generics! (simplifies parsing)

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

### Parameters

Parameters in this language are a unification of generics and normal parameters in other languages. They're used for both types and functions. (and modules if we decide to add them)

Goals:
* types and values can be freely mixed
* parameters and types are _immediately_ in-scope for the following parameters

Implementation:

### Language semantics

First thoughts:
* Everything (appears to be) garbage collected and object-oriented? (like python)
* Lists etc don't use pass my copy, also just like python.
* What about mutability? Eg. functions writing to their parameters? Do we just ban this entirely? We kind of want this for unit testing at least.
  * Maybe classes exist and behave differently?
  * Or some specific type for shared stuff, eg. `&` or `Ptr(T)`?
* Do we use Rust reference semantics but with garbage collecting?

### Data structures

list: backed by a dequeue 
  * remove(index), pop/push both sides (back is implicit), len, ...
array: simple fixed len array
ranges: self-explanatory
option: unwrap

### Literals

```
true, false
0, 1, 0xA
```

### Clock specifier

How to specify the clock values are synchronized too?

Should the clocking be part of the type or of the specific values? Maybe have clocks be implicit by default, but with some option to specify them?

`def flip_flop(clk: Clock, x: Bool[clk])`
or
`def flip_flop(clk: Clock, x[clk]: Bool)`

Either way, if there's only one clock we'll assume that one by default:
`def flip_flop(clk: Clock, x: Bool)`

### Type conversion

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

### Build system

### IDE plugins

### Package manager

