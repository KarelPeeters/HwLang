# Invalid Domain Crossing — lyrics v4

[Intro — spoken, robotic, over a ticking clock]
error: invalid domain crossing.
async... to sync.

[Verse 1]
Three a.m. in the waveform viewer, staring at a sum gone wrong:
`a + b` lost the carry, and it's been that way all along.
Verilog just dropped the top bit, didn't even raise a flag:
"context-determined width", it says, like that's a feature, not a gag.
Mix a signed with an unsigned and the whole expression flips its sign;
plug a four-bit wire in an eight-bit port, and the tools all say it's fine.
So I fled to VHDL, where the types are carved in stone —
four conversions deep to get a bit I'm allowed to call my own.
(`std_logic_vector`, `unsigned`, `to_integer`, `resize`...)
I left one signal off the sensitivity list, and now
the sim says zero, the board says one, and nobody knows how,
and a missing `else` in a `case` I wrote last May
grew a latch inside my netlist, and it's living there today.

[Pre-Chorus 1]
Blocking or non-blocking — which arrow goes where?
Is it a wire or a flop, and does the tool even care?
Three months before tape-out the CDC tool finally ran:
forty-one violations and a freshly slipping plan.

[Chorus]
No more latches, no sensitivity lists,
one equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"invalid domain crossing!" — and I haven't shipped it yet!
K! H! D! L! Count the values, not the bits.
K! H! D! L! If it compiles, then it fits.

[Verse 2 — range types]
I don't count the bits no more, I write the values that I need;
the compiler does the `$clog2` math at elaboration speed.
Add two numbers, keep the carry; squeeze it narrow? `type mismatch`, friend —
if you want the counter wrapping, write the modulo at the end.
No casting done behind my back, and no four-deep cast parade:
want the bits? `to_bits`, and back again — the round trip's already made.
A one-flop synchronizer won't type-check anymore:
two flops minimum, in the type — that's what range types are for.

[Pre-Chorus 2]
`comb` or `clocked` — only two kinds of block.
Want a flop? Then say `reg wire`; it won't sneak in on the clock.
Skip the reset? Write `= undef` and say it out loud:
I didn't forget it — I declared it, and I'm proud.

[Chorus]

[Verse 3 — interfaces]
Remember wiring up a bus? Two hundred ports, line by line,
`.valid(valid)`, `.ready(ready)` — one got swapped, the lint says "fine."
Now I write the `interface` once and give each side a `view`:
`input` here and `output` there, and the compiler checks them too.
Every stream is one port now, with its clock and type inside;
the top level reads like the block diagram, nothing left to hide.
The handshake lives behind an API you can't misuse,
so the review is about the design, not a thousand lines of glue.

[Bridge — rapid talk-sung, like reading a compiler log]
"Driver mismatch!" — that's a latch you'll never meet.
"Self-loop!" — read before you drive? Take a seat.
"Multiple drivers!" — two blocks fighting on one wire.
"Different clock!" — caught before the chip's on fire.
No Perl that writes the Verilog, no Python writing Perl,
no Jinja for the port lists, no Tcl to glue the world,
no homegrown lint in a nightly cron, no build script held with tape —
types are values, modules are functions, and it all compiles in one take.

[Breakdown — half-time, sparse]
And when I really have to cross,
I'll do it — but I'll sign:
`unsafe_value_with_domain`...
grep knows every line.

[Verse 4 — compile speed]
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
No more latches, no sensitivity lists,
one equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset —
"invalid domain crossing!" — and I haven't shipped it yet!
K! H! D! L! Count the values, not the bits.
K! H! D! L! If it compiles...
K! H! D! L! If it compiles, then it fits!

[Outro — spoken, robotic, fading]
khdl build: finished in twenty milliseconds.
zero errors. zero warnings.
count the values... not the bits.
