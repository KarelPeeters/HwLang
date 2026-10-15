# Invalid Domain Crossing: storyboard

Track: `../audio/khdl.wav`, 331.18 s, about 129 BPM. Lyric timing: `../analysis/lyrics.json` (Whisper aligned to `suno.md`).

## Visual identity

- **Medium:** a single 1920×1080 Canvas 2D. Every frame is a pure function of `t`, so headless Chrome renders it deterministically.
- **Look:** "EDA tool at 3 a.m., reimagined as a design object". Near-black ink background (#0a0c11) with a faint
  schematic dot grid. Thin hairline rules, like a datasheet. Big heavy grotesk kinetic type (Archivo Black) for the hook
  words, JetBrains Mono for anything that is code, and Space Grotesk for labels.
- **Colour has meaning.** Each clock domain has its own colour, the way KHDL gives every value a domain:
  - `clk_a` cyan #3ce0ff, `clk_b` magenta #ff4fd8, `async` amber #ffb23e;
  - error red #ff3b5c, ok green #4cf2a0, text #e9edf4, muted #6b7385.
- **Motion language:**
  - everything snaps to the beat (kick envelope from `analysis/audio.json`);
  - hard-edged wipes like a logic transition: 0 to 1 in one frame plus a short ringing;
  - words "clock in" on their sung onset, like a flop latching data.
- **Real artefacts:** the source, diagnostics and Verilog shown on screen come from the real compiler.
  - The page loads `khdl_wasm` and calls `run_all()` while code is being "typed", so squiggles and errors appear the way
    they really would.
  - Syntax highlighting uses the compiler's own tokenizer (`codemirror_tokenize_to_tree`).
  - The playground scene uses the real playground (`build/web_demo`) captured as frames.

## Scenes

| t (s) | Section | Visual |
|---|---|---|
| 0 – 2.7 | Intro | A black screen. A single clock waveform draws itself left to right, ticking with the beat. |
| 2.7 – 11 | Intro, spoken | Terminal cursor. The real `invalid domain crossing` diagnostic types out in full colour. Two waveform lanes in different colours (`async`, `sync(clk)`) drift out of phase and glitch where they meet. |
| 11 – 26.7 | Intro, instrumental | Title card build: "INVALID DOMAIN CROSSING" stamped on the beats. The letters K H D L arrive as four flops in a shift register, one per beat. The camera pulls back through a netlist. |
| 26.7 – 41 | V1 | **Waveform viewer at 3 a.m.** GTKWave-style lanes for `a`, `b`, `sum`. The carry bit visibly falls off the top of `sum` into the void. A sticky-note label reads `context-determined width` with a mock "feature" badge. |
| 41 – 48.5 | V1 | **Sensitivity list.** `always @(a or b)` with `c` missing. A split screen shows "SIM: 0" against "BOARD: 1" flickering. |
| 48.5 – 55.9 | V1 | **The latch grows.** A `case` with a missing `else`. A latch symbol sprouts in the netlist like an organism, with vines of wires. |
| 55.9 – 63.8 | V1 | **VHDL, three times.** The same port list is stamped as `entity`, then `component`, then `port map`, stacking and scrolling until it overflows the screen. |
| 63.8 – 71 | PC1 | **`=` vs `<=`.** Two arrows spin like a slot machine. "wire? flop? latch?" flashes in an `logic` type box. |
| 71 – 82.4 | PC1 | **CDC tool.** A progress bar finally completes, the violations counter races to 41, and a Gantt chart slips to the right, bar by bar. |
| 82.4 – 111 | Chorus 1 | Big kinetic type over a field of clock-domain lanes. "No more latches" crosses out a latch glyph. The `=` sign is huge. Clock and reset icons attach to wires. "Crossing" triggers the real error card. **K! H! D! L!** hits on the beats as full-screen letters. "Range → bits" is a `int(0..256)` → `[7:0]` morph. "If it compiles, then it fits": blocks drop into slots, Tetris-like, then clear. The instrumental tail is a fly-through of the domain lanes. |
| 111.4 – 134.4 | V2 | **Range types.** A number line. Ranges are intervals: `0..8 + 0..8` slide and fuse into `0..15`. Squeezing into a narrower slot triggers the real `type mismatch` diagnostic. A modulo wrap is a circle. The synchronizer is a chain of flops, and the type `int(2..)` rejects a chain of one. |
| 134.4 – 146 | PC2 | **Two kinds of block.** `comb {}` and `clocked {}` as two clean shapes. `reg wire` stamps a flop onto a wire. `= undef` is proud: an "X" with a badge. |
| 146 – 168 | Chorus 2 | The chorus system again, in a new colourway (magenta domain), plus code tiles. |
| 168.9 – 182.6 | V3 | **Port-list spaghetti.** Two hundred wires between two blocks; one pair swaps colour, and lint says "fine ✓". Then they all collapse into one thick bus, `axi_stream`, with `view input` and `view output` labels at each end. |
| 182.6 – 197.2 | V3 | **The block diagram is the code.** A real top-level KHDL snippet on the left, and on the right the block diagram it draws itself into. The API "lock" on the handshake. |
| 197.2 – 207.3 | Bridge | **Compiler log rap.** Four real diagnostics slam in on each hook ("Driver mismatch!", "Self-loop!", "Multiple drivers!", "Different clock!"). |
| 207.3 – 220.5 | Bridge | **The tool graveyard.** Files `gen_ports.pl`, `codegen.py`, `ports.j2`, `build.tcl`, `lint_nightly.cron`, `Makefile` appear, each struck through as it is sung. The finale is "types are values, modules are functions": one `.kh` file. |
| 221 – 229 | Breakdown | Nearly empty. A single line of `unsafe_value_with_domain` glows, and a `grep -rn` result list lights each hit. |
| 229 – 243.6 | V4 | **Speed.** A 32-core heatmap lights up per module. The editor shows squiggles appearing live, using the real compiler on typed code. The license-server spinner never finishes ("0 seats available"). |
| 243.6 – 258.4 | V4 | **Tests.** pytest dots race across the screen and "380 passed in 4.0s". A Rust-style caret points at the "crime". |
| 258.4 – 274.3 | PC3 | **Verilog 2005 out.** KHDL source on the left morphs into the real generated Verilog on the right, which then flows into generic tool boxes (sim, synth, lint, LEC, P&R) that all light up green. |
| 274.3 – 283.9 | Key change | The whole frame shifts colourway (all domains rotate hue) and a playground tour plays: frames of the real web demo. |
| 283.9 – 310 | Final chorus | The chorus system at maximum: every element from the song is reprised in the background tiles. |
| 310 – 331 | Outro | Terminal: `$ khdl build`, `finished in 20 ms`, `0 errors, 0 warnings`. Then "if it compiles… it fits." and the logo fades to a single clock waveform, which stops. |

## Lyrics layer

- The current line is set in large type in the lower third, or centred in chorus scenes. Each word clocks in on its sung
  onset.
- Code phrases are shown as KHDL code in mono, for example `reg wire`, `= undef`, `$clog2`, `unsafe_value_with_domain`
  and `a + b`.
