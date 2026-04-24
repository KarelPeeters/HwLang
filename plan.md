# Plan: C++ simulator wrapper + recording data store + egui waveform GUI prototype

## Problem & approach

We want a working end-to-end prototype of:

1. A **C++-backend simulator wrapper** that mirrors the Verilator wrapper's API
   (ports getters/setters, `step(delta)`), built on top of the existing
   `lower_to_cpp` codegen. The wrapper compiles the generated C++ to a shared
   library and dlopens it. For now `step` calls the top module's
   `module_<idx>_all` function 32 times (no convergence detection yet).
2. **Python integration** mirroring `as_verilated` / `VerilatedInstance` so the
   existing compare tests can additionally compare against the C++ simulator.
3. An **in-memory waveform data store** that records every signal (ports + all
   wires, recursively across the module hierarchy) at every step, on the Rust
   side.
4. An **egui-based GUI** (eframe + glow backend) that:
   - Embeds the simulator live (run/step, see new samples appear)
   - Can also load a previously serialized data store
   - Browses the module hierarchy as a tree
   - Lets the user add/remove/reorder signals in a waveform pane
   - Provides scroll+zoom on the time axis (time runs left-to-right)
   - Renders multi-bit signals (arrays, tuples, structs) as **collapsible
     rows** so users can expand/inspect element- or field-wise traces

All Rust prototype code lives in `hwl_language` for now. The egui GUI is gated
behind a new optional cargo feature so existing builds aren't impacted.

## Architecture

The implementation is structured around five new pieces:

1. **`hwl_common::back::lower_cpp_wrap`** (new module, sibling of
   `lower_verilator`): generates a thin C ABI wrapper around the C++ produced
   by `lower_to_cpp`. Exposes:
   - `check_hash()` for runtime safety
   - `create_instance() / destroy_instance()`
   - `step(instance, increment_time)` — calls `module_<top>_all` 32 times,
     swapping prev/next signal+ports buffers
   - `get_port(instance, idx, len, ptr) / set_port(...)`
   - `get_signal(instance, signal_id, len, ptr)` for hierarchical wire reads
     used by the recorder
   - A static signal table describing every (module-instance-path, signal-name,
     bit-width, IrType) the recorder can enumerate at startup.

2. **`hwl_language::back::wrap_cpp`** (new, gated behind `wrap_cpp` feature
   like `wrap_verilator`): mirrors `wrap_verilator.rs`. Provides
   `CppSimLib` / `CppSimInstance` with the same `step / get_port / set_port`
   surface and the same `SimulationFinished` enum. Adds `enumerate_signals()`
   and `get_signal()` for hierarchy-aware reads.

3. **`hwl_language::sim::recorder`** (new module): a `WaveStore` data structure
   that holds, per signal, a vector of `(time, bits)` change events using
   delta-compression (only changes are stored). `WaveStore::sample(instance,
   time)` snapshots all signals from a `CppSimInstance`. Serializable via
   serde (JSON for now; FST/VCD export can come later).

4. **`hwl_python` additions**: new classes `ModuleCpp` (analogous to
   `ModuleVerilated`) and `CppInstance` (analogous to `VerilatedInstance`) with
   the same Python-facing API: `.ports`, `.step(delta)`. Compile/link the C++
   via `g++ -shared -O0 -fPIC` invoked through `Command`.

5. **`hwl_language::bin::wave_gui`** (new binary, behind a `gui` feature):
   eframe app with two panes:
   - Left: hierarchy tree (`egui::CollapsingHeader`)
   - Right: waveform viewer — a single horizontally-scrolling/zooming pane
     where each row has, on its **left**, the signal name + current value
     label, and on its **right**, the waveform itself drawn left-to-right
     along the time axis. Multi-bit composite signals (arrays, tuples,
     structs) render as collapsible parent rows whose children are the
     element/field sub-signals.

## Compilation pipeline for the C++ backend

Mirrors what `as_verilated` does, simpler:
1. `lower_to_cpp(diags, modules, [top])` -> C++ string
2. `lower_cpp_wrap(modules, top, &cpp_source)` -> wrapper C++ + check_hash
3. Write both to `build_dir/lowered.cpp` and `build_dir/wrapper.cpp`
4. `g++ -O0 -fPIC -shared -std=c++17 lowered.cpp wrapper.cpp -o sim.so`
5. `unsafe { CppSimLib::new(&modules, top, check_hash, &path_so) }`

`-O0` keeps prototype iteration fast; can be raised later.

## Testing plan

- Extend `hwl_sandbox.common.compare.CompiledCompare` so it holds both a
  `VerilatedInstance` and a `CppInstance` and runs/asserts both in `eval`. Do
  **not** skip or guard around C++-backend failures — run everything; failing
  cases will be triaged by the user afterwards.
- Add a Python test for the C++ backend that exercises a **clocked-process
  module**, modeled on the existing `tests/module/test_reg.py` and
  `tests/flow/test_var_signal.py` clocked tests, instantiating via
  `Module.as_cpp(build_dir).instance()` and stepping the clock.

## GUI prototype scope (intentionally minimal)

In:
- Hierarchy tree view of module instances + their signals
- Add/remove/reorder waveform rows
- Horizontal scroll + zoom on time axis (time runs left-to-right)
- Per-row name + current-value label on the left of each waveform row
- Multi-bit value formatting for arrays / tuples / structs as
  **collapsible rows** whose children are element/field sub-traces
- "Step N" / "Run to time T" buttons
- "Save / Load store" buttons (JSON)

Out (deliberately deferred):
- Cursors/markers, search, regex filtering
- VCD/FST import/export
- Convergence detection in `step`
- LOD / GPU-accelerated waveform rendering (just draw transitions directly
  with `egui::Painter` for now)

## Notes / risks

- The 32-iteration fixed `step` loop is wrong in general (no convergence
  detection, no sensitivity tracking) — explicitly called out as TODO.
- `lower_to_cpp` already errors on external modules and on some expressions/
  types. The compare tests must tolerate these failures by detecting the
  diagnostic and skipping the C++ leg of the comparison.
- The recorder snapshots **every** signal on every step. For long runs this
  will be memory-heavy; acceptable for prototype, flagged as TODO.
- Putting the GUI binary in `hwl_language` pulls eframe/egui into that crate's
  feature graph. Mitigated by gating it behind a `gui` feature so default
  builds (LSP, wasm, python) are unaffected.

## Todos

Tracked in SQL (see todos table). High-level ordering:

1. C++ wrapper codegen (`lower_cpp_wrap`)
2. Rust C++ sim wrapper (`wrap_cpp`)
3. Build/link helper + Rust smoke test
4. Recorder data store
5. PyO3 `ModuleCpp` / `CppInstance` bindings
6. Update `compare.py` to also check C++ backend
7. eframe binary skeleton + hierarchy tree
8. Waveform pane (rendering + scroll/zoom)
9. Add/remove/reorder signals UI
10. Live-step controls + load/save store
