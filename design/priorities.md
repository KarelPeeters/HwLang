# Project config

* add hwl.toml config file
  * contains file search paths including which import paths that map to (std as a default)
  * contains misc settings for formatter, lsp, codegen, ...
  * _maybe_ contains default top?
  * commandline tool and lsp look for this file and base their view of the project on it
  * this also allows multiple projects in a single lsp instance

# Frontend features

* add "elaboration trace" to all error messages, to know where the error actually happened

* range fixes (ensure always increasing, never crashing)
  * also allow non-contiguous ranges, switch to non-inclusive ranges, add int range tracking to match
  * implication fixes (if empty drop the block instead of stopping)

* allow hardware return/break/continue
* pass expected type to struct constructors
* add functions to structs/enums/interfaces
  * static (without self param), used for constructors and to replicate generic packages
  * non-static (with self param)
  * with `inout`/`ref` self param, to mutate it in-place

* for utilities, we need something more more convenient than a module and more powerful then an interface + functions
  * or is the "reg" trick inside functions enough?

# Simulation

* get C++ backend to be as convenient as verilator
* add pipe backend for use with legacy simulators
* write some good testbench examples and test cases to add to CI

# Backend fixes

* fix verilog backend
  * array/tuple indexing, bit widths, signedness, zero-width signals, ...
* proper Python/Rust/C++ simulation setup
* proper Python/Rust/Verilator simulation setup

* fuzz testing for expressions?

# LSP

* autocomplete
* auto-import
* auto formatter (not only LSP, also CLI)

# Performance

* write profile info to firefox profile for easy viewing: https://crates.io/crates/fxprof-processed-profile

# Docs

* write some proper examples, eg. RISCV CPU, some state machine thing, some DSP, some generic utility blocks
* write docs, some markdown book thing ideally similar to Rust/Zig/readthedocs
