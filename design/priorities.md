# Frontend features

* range fixes (ensure always increasing, never crashing)
  * also allow non-contiguous ranges, switch to non-inclusive ranges, add int range tracking to match
  * implication fixes (if empty drop the block instead of stopping)
* add something to cast domains
* allow type inference for vars/vals in more cases, eg. array literals should just work
* allow hardware return/break/continue

* external modules
* some way to configure constants, used for ram models vs cells
* better top module specification

* for utilities, we need something more more convenient than a module and more powerful then an interface + functions
  * or is the "reg" trick inside functions enough?

# Backend fixes

* fix verilog backend
  * array/tuple indexing, bit widths, signedness, zero-width signals, ...
* proper Python/Rust/C++ simulation setup
* proper Python/Rust/Verilator simulation setup

* fuzz testing for expressions?

# LSP

* LSP autocomplete
* LSP auto-import
* (LSP) auto formatter

# Performance

* add Arc to all potentially expensive values and hope for a free performance win

# Docs

* write some proper examples, eg. RISCV CPU, some state machine thing, some DSP, some generic utility blocks
* write docs, some markdown book thing ideally similar to Rust/Zig/readthedocs
