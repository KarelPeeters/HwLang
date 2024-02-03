## Design guidelines

* Module/Functions/Structures/... should typecheck on their own, independent of their users.
    * This means that an error is _either_ inside of something or at the use site. It can't propagate outwards.
    * Unlike C++ template parameters!
* Prefer stdlib over language features
* Privilege the stdlib as little as possible, libraries should be able to do the same
* Put enough semantics into the language to avoid any possible backend errors.
  * Examples: clocks should be marked as such, values should know which clock if any they are synced too.
  * No delta cycles!

## Specifics

### Types

Prefer types defined in the stdlib over built-in types.

Types:
```
Bool
Int, UInt, Int(min=..., max=...)
B(n), I(n), U(n) -> B8, I8, U8
IeeeFloat(signed, mantissa, exponent) -> F32, F64
FixedPoint(signed, integral, fractional) -> UF[4, 3], IF[4, 3] 
```

Do we want types to be first-class? Probably best, so we can just pass them as ordinary arguments. This means that type constructors should also just be functions.

### Values

### Coding conventions

Capitalization:
* Types: upper camel
* Constants: upper snake
  * What about generics? They're going to look a lot like parameters!
* Everything else: lower camel or lower snake?