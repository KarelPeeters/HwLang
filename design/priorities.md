# Frontend features

* match

* infer domain for reg (based on first usage) and wire (based on init expr)
* range fixes (ensure always increasing, never crashing)
* implication fixes (if empty drop the block instead of stopping)
* add something to cast domains
* fix type inference for vars/vals
* allow hardware return/break/continue

# Backend fixes

* fix verilog backend
* proper Python/Rust/C++ simulation setup
* proper Python/Rust/Verilator simulation setup

# LSP

* LSP autocomplete
* LSP auto-import
* (LSP) auto formatter

# Performance

* add Arc to all potentially expensive values and hope for a free performance win
