# Basics

* python wrapper for input/output
* process sensitivity lists
* process scheduling
* implement undef
* document undef semantics (ie. don't assume anything about the value part of things that are marked undef)
* add global context parameter, containing assertion and print callbacks
* record signal changes into convenient database
  * queryable from python?
  * will serve as the base for the GUI

# Optimization

* parallelize
  * module lowering
  * process lowering?
  * llvm function optimization and compilation
* enable process optimization
* fuse sequential processes?
* on-disk compilation cache
* allow simulator save/restore
* first run with un-optimized functions, then gradually swap them in as they complete
* PGO and re-optimize?
* implement rare things (X/Z/assertions/prints?) with a callback to Rust to reduce code size?
* for writes to large arrays 
* sparse process calling:
  * only call clocked blocks if they actually have a reset or clocked edge, this is easy to implement and forms a nice batch
  * only call comb blocks if one of their inputs changed, this is trickier

# Fancy features

* add accessors and force override for intermediate signals
* add GUI
* add backdoor read: pretty simple, just read the right bits in the state buffer
* add backdoor write (force): additional stage after every signal that writes to forced signals,
  that re-sets the value to the forced one

# Resources

* https://llvm.org/devmtg/2023-10/slides/techtalks/Erhart-Arcilator-FastAndCycleAccurateHardwareSimulationInCIRCT.pdf
* https://rcor.me/papers/cgo22rolag.pdf
* https://github.com/verilator/verilator

# Undef design details

* careful for ints: they can go out of range (if range is not exact binary range)
* repr: if undef bit is set, value can be anything. This makes impl fast and simple. Implementations of operations have
  to make sure they accept any (even invalid) values. For arith this is easy, for things like indexing we have to be
  extra careful. 
* make make the bit mean "defined", so we can just zero-init everything for the initial setup