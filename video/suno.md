# Suno input (lyrics v5)

## Title
Invalid Domain Crossing

## Style (paste into "Style of Music")
driving synth-pop electro-rock, 124 bpm, punchy analog synth bass, gated 80s drums, arpeggiated sequencer, confident male vocal, talk-sung rapid bridge, huge anthemic chorus, key change final chorus, clean modern mix

## Lyrics (paste into "Lyrics")

[Intro]
[Spoken, robotic]
Error: invalid domain crossing.
Async... to sync.

[Verse 1]
Three a.m. in the waveform viewer, staring at a sum gone wrong:
A plus B lost the carry, and it's been that way all along.
Verilog just dropped the top bit, didn't even raise a flag,
"Context-determined width," it says, like that's a feature, not a gag.
I left one signal off the sensitivity list, and now
The sim says zero, the board says one, and nobody knows how,
And a missing else in a case I wrote last May
Grew a latch inside my netlist, and it's living there today.
So I tried VHDL instead, where you say it all three times:
Entity, component, port map, and it's longer than these rhymes.

[Pre-Chorus]
Blocking or non-blocking, which arrow goes where?
Is it a wire or a flop, and does the tool even care?
Three months before tape-out the CDC tool finally ran:
Forty-one violations and a freshly slipping plan.

[Chorus]
No more latches, no sensitivity lists,
One equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset,
And the compiler catches every crossing I forget!
K! H! D! L! You give the range, it picks the bits.
K! H! D! L! If it compiles, then it fits.

[Verse 2]
I don't count the bits no more, I write the values that I need;
The compiler does the c-log-two math at elaboration speed.
Add two numbers, keep the carry; squeeze it narrow? Type mismatch, friend,
If you want the counter wrapping, write the modulo at the end.
A one-flop synchronizer won't type-check anymore:
Two flops minimum, in the type, that's what range types are for.

[Pre-Chorus]
Comb or clocked, only two kinds of block.
Want a flop? Then say "reg wire", it won't sneak in on the clock.
Skip the reset? Write "equals undef" and say it out loud:
I didn't forget it, I declared it, and I'm proud.

[Chorus]
No more latches, no sensitivity lists,
One equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset,
And the compiler catches every crossing I forget!
K! H! D! L! You give the range, it picks the bits.
K! H! D! L! If it compiles, then it fits.

[Verse 3]
Remember wiring up a bus? Two hundred ports, line by line,
Dot valid valid, dot ready ready, one got swapped, the lint says "fine."
Now I write the interface once and give each side a view:
Input here and output there, and the compiler checks them too.
Every stream is one port now, with its clock and type inside;
The top level reads like the block diagram, nothing left to hide.
The handshake lives behind an API you can't misuse,
So the review is about the design, not a thousand lines of glue.

[Bridge]
[Rapid talk-sung, like reading a compiler log]
"Driver mismatch!" That's a latch you'll never meet.
"Self-loop!" Read before you drive? Take a seat.
"Multiple drivers!" Two blocks fighting on one wire.
"Different clock!" Caught before the chip's on fire.
No Perl that writes the Verilog, no Python writing Perl,
No Jinja for the port lists, no Tcl to glue the world,
No homegrown lint in a nightly cron, no build script held with tape,
Types are values, modules are functions, and it all compiles in one take.

[Breakdown]
[Half-time, sparse, intimate]
And when I really have to cross,
I'll do it, but I'll sign:
Unsafe value with domain...
Grep knows every line.

[Verse 4]
Hit save, and every core lights up, every module on its thread,
The squiggles hit my editor before the thought has left my head.
Elaborated, type-checked, Verilog written, all of it done
Before the license server's even checked if there's a seat. There's none.
Four lines of Python for a testbench, pytest on the side,
Three hundred eighty tests in four seconds, Verilator for the ride.
Rust-style errors with a caret pointing right at my crime,
Your linter's three months late; my compiler's right on time.

[Pre-Chorus]
It all comes out as Verilog two-thousand-five,
Every tool you've ever cursed will keep it alive.
No waiting ten more years for the vendors to parse a struct,
The language lives in the compiler, and we're finally unstuck.

[Key Change]

[Final Chorus]
No more latches, no sensitivity lists,
One equals sign, and it does just what it says it is.
Every wire knows its clock, every flop knows its reset,
And the compiler catches every crossing I forget!
K! H! D! L! You give the range, it picks the bits.
K! H! D! L! If it compiles...
K! H! D! L! If it compiles, then it fits!

[Outro]
[Spoken, robotic, fading]
K H D L build: finished in twenty milliseconds.
Zero errors. Zero warnings.
If it compiles... it fits.

[End]

## Tips
- The song runs about 4.5 minutes. Use a recent Suno model (v4.5 or later) so it doesn't cut off. If it still ends early,
  use "Extend" from the last finished section, or drop Pre-Chorus 3.
- Generate several takes. Pick by ear for: crisp error strings in the bridge, the "K! H! D! L!" chant, and a clear key change.
- If the style field is limited to 200 characters, drop "clean modern mix" and "arpeggiated sequencer" first.
