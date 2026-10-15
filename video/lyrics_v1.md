# No Delta Cycles — lyrics draft v1

Style prompt (for a music model): driving synth-pop / electro-rock, 124 BPM, punchy analog bass,
gated drums, confident male-or-female lead with a talk-sung bridge, big anthemic chorus, ~3:30.

---

[Intro — spoken, over a ticking clock]
error: invalid domain crossing.
async... to sync.

[Verse 1]
Three a.m. in the waveform viewer, cursor parked upon a glitch,
Sixteen bits went in the adder and the carry made a switch —
`assign sum = a + b`, and not a single warning came,
Verilog just sliced the MSB and called it "context width" by name.
I forgot one signal in the sensitivity list again,
Now the sim says one and synthesis says ten,
And a missing `else` in a `case` I wrote in May
Grew a latch inside my netlist and it's living there today.

[Pre-Chorus]
Blocking or non-blocking — which arrow goes where?
Delta cycles spinning like a VHDL prayer,
Three months before tape-out, the CDC tool finally ran:
Forty-one violations, and a meeting with the man.

[Chorus]
No delta cycles, no sensitivity lists,
One equals sign and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"Invalid domain crossing", and I haven't shipped it yet!
K-H-D-L — count the values, not the bits,
K-H-D-L — if it compiles, then it fits.

[Verse 2]
`int(0..256)`, I don't do `$clog2` by hand,
Add two `uint(0..8)`s, get `0..15` — the compiler understands.
Squeeze it into something narrower? That's a type mismatch, friend,
If you want the counter wrapping, write the `% 2**32` at the end.
`if (p < 4)`, and in that branch the range of `p` shrinks down,
Divide by something that could be zero? Not in this town.
A one-flop synchronizer? `N: int(2..)` says no,
The type system read the ZipCPU blog a long time ago.

[Pre-Chorus 2]
`comb` or `clocked` — there's no third kind of block,
Want a flop? Then say `reg wire`, no register by shock.
Skip the reset? Write `= undef` and own it out loud —
I didn't forget it, I declared it, and I'm proud.

[Chorus]
No delta cycles, no sensitivity lists,
One equals sign and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"Invalid domain crossing", and I haven't shipped it yet!
K-H-D-L — count the values, not the bits,
K-H-D-L — if it compiles, then it fits.

[Bridge — rapid talk-sung, like reading a compiler log]
`driver mismatch between conditional branches` —
that's a latch you'll never meet.
`combinatorial self-loop` —
read before you drive? Take a seat.
`clocked process cannot drive port without marking it as a register` —
yeah, the error's long, but your bring-up got a whole lot better.
No `generate`, no `genvar`, no Perl, no Jinja template,
just a `for` loop and `ident("x_{i}")` to name it.
Generic modules? They're functions that return modules,
types are values, packages too — I know, it's unusual.

[Breakdown — half-time, sparse]
And when I really need to cross,
I sign my name in blood:
`unsafe_value_with_domain`...
grep finds every flood.

[Verse 3]
It all comes out as Verilog two-thousand-five,
Every vendor tool you've ever cursed will keep it alive.
No waiting ten years for the EDA to parse a struct,
The language moves in the compiler — the vendors are unstuck.
Testbench in four lines of Python, pytest on the side,
Three hundred eighty tests in four seconds, Verilator for the ride,
Rust-style errors with a caret pointing at your crime —
Your linter's three months late; my compiler's right on time.

[Final Chorus — key change]
No delta cycles, no sensitivity lists,
One equals sign and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"Invalid domain crossing", and I haven't shipped it yet!
K-H-D-L — count the values, not the bits,
K-H-D-L — if it compiles...
K-H-D-L — if it compiles, then it fits.

[Outro — spoken, fading, over the ticking clock]
khdl build... finished.
zero errors. zero warnings.
no delta cycles.
