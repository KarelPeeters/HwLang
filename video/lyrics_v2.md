# No Delta Cycles — lyrics v2

[Intro — spoken, robotic, over a ticking clock]
error: invalid domain crossing.
async... to sync.

[Verse 1]
Three a.m. in the waveform viewer, cursor parked upon a glitch,
Two eight-bit numbers walked into an adder — eight bits made it out, that's it.
Verilog just dropped the carry, didn't even raise a flag:
"context-determined width", it says, like that's a feature, not a gag.
I left one signal off the sensitivity list, and now
the sim says zero, the board says one, and nobody knows how,
and a missing `else` in a `case` I wrote last May
grew a latch inside my netlist, and it's living there today.

[Pre-Chorus 1]
Blocking or non-blocking — which arrow goes where?
Delta cycles spinning like a VHDL prayer.
Three months before tape-out the CDC tool finally ran:
forty-one violations and a freshly slipping plan.

[Chorus]
No delta cycles, no sensitivity lists,
one equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"invalid domain crossing!" — and I haven't shipped it yet!
K! H! D! L! Count the values, not the bits.
K! H! D! L! If it compiles, then it fits.

[Verse 2]
`int(0..256)` — no `$clog2` by hand,
add two `uint(0..8)`, get `0..15`, the compiler understands.
Squeeze it into something narrower? That's a `type mismatch`, friend;
if you want the counter wrapping, write the `% 2**32` at the end.
`if (p < 4)` — and in that branch the range of `p` shrinks down;
divide by something that could be zero? Not in this town.
A one-flop synchronizer won't type-check anymore:
`N: int(2..)` — that's what range types are for.

[Pre-Chorus 2]
`comb` or `clocked` — only two kinds of block.
Want a flop? Then say `reg wire`; it won't sneak in on the clock.
Skip the reset? Write `= undef` and say it out loud:
I didn't forget it — I declared it, and I'm proud.

[Chorus]

[Bridge — rapid talk-sung, like reading a compiler log]
`driver mismatch between conditional branches` —
that's a latch you'll never meet.
`combinatorial self-loop` —
read before you drive? Take a seat.
`clocked process cannot drive port without marking it as a register` —
yeah, the error's long, but your bring-up got a whole lot better.
No `generate`, no `genvar`, no Perl, no Jinja template,
just a `for` loop and an `ident("x_{i}")` to name it.
Types are values, modules are functions,
compile-time code without the dysfunction.

[Breakdown — half-time, sparse]
And when I really have to cross,
I'll do it — but I'll sign:
`unsafe_value_with_domain`...
grep knows every line.

[Verse 3 — compile speed]
Hit save, and every core lights up, every module on its thread,
the squiggles hit my editor before the thought has left my head.
Elaborated, type-checked, Verilog written — all of it done
before the license server's even checked if there's a seat. (There's none.)
Four lines of Python for a testbench, pytest on the side,
three hundred eighty tests in four seconds, Verilator for the ride.
Rust-style errors with a caret pointing right at my crime —
your linter's three months late; my compiler's right on time.

[Pre-Chorus 3]
It all comes out as Verilog two-thousand-five,
every tool you've ever cursed will keep it alive.
No waiting ten more years for the vendors to parse a struct —
the language lives in the compiler, and we're finally unstuck.

[Key change]

[Final Chorus]
No delta cycles, no sensitivity lists,
one equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"invalid domain crossing!" — and I haven't shipped it yet!
K! H! D! L! Count the values, not the bits.
K! H! D! L! If it compiles...
K! H! D! L! If it compiles, then it fits!

[Outro — spoken, robotic, fading]
khdl build: finished in twenty milliseconds.
zero errors. zero warnings.
no delta cycles.
