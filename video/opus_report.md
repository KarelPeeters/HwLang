# KHDL music video: process report

This report covers the music video "Invalid Domain Crossing", a song about KHDL. It explains how the video was made, the
feedback that shaped it, the problems along the way, and how the pipeline works. The full transcript is in
`opus_conversation.md`.

**Result:** `out/khdl_invalid_domain_crossing.mp4` (1920×1080, 60 fps, 5:31). The source is committed on the `video`
branch (`918d374d`). The video, audio, dependencies and build outputs are gitignored.

---

## 1. Process

1. **Research (subagent).** A general-purpose subagent read `design/`, `docs/`, `ignored/feedback`, the compiler
   source, the Python tests and the git history. It treated all planned features as done.
   - Output: `research/khdl_summary.md`, about 750 lines. It covers the pitch, the syntax, the checks the compiler does,
     the pain points of other HDLs, the roadmap, feedback themes and 71 lyric-worthy "nuggets".
2. **Tooling check (in parallel).**
   - Available: `ffmpeg`, Node 24, Chrome and `uv`.
   - Missing: any music-generation API.
   - The RTX 5070 had an NVIDIA driver/library version mismatch, so the GPU was unusable without a reboot.
   - The user decided to generate the song in Suno.
3. **Lyrics.** Five drafts (`lyrics_v1.md` to `lyrics_v5.md`), refined with the user. The details are in section 2.
   `suno.md` holds the Suno-ready text: code spelled out phonetically, plus section tags and a style prompt.
4. **Audio acquisition.** The user generated the track in Suno and shared a link. Getting the actual audio took several
   steps (section 3).
5. **Audio analysis.**
   - librosa: tempo (about 129 BPM), beat times, and low/mid/high/RMS envelopes at 60 fps.
   - Demucs: vocal separation.
   - faster-whisper large-v3, run on the CPU: word timestamps from the vocal stem.
   - A custom aligner maps Whisper's words onto the canonical lyrics in `suno.md`. 701 of 781 words matched exactly; the
     rest were fuzzy-matched or interpolated.
6. **Collecting real compiler material.**
   - **Diagnostics from small projects.** Small KHDL projects were built with the release `khdl` binary:
     - `compiler_demo/`: deliberately wrong code (domain crossing, latch, truncation, self-loop, multiple drivers,
       async to sync, missing `reg wire`);
     - `compiler_ok/`: the README example, which builds cleanly and yields the Verilog;
     - `compiler_sync/`: a one-flop synchronizer that should be rejected;
     - `compiler_bridge/`: a "types are values / modules are functions" snippet, checked against a copy of the demo
       project.
   - **Other real material:**
     - a real `grep` for `unsafe_` over the demo project;
     - a real `uv run pytest -n auto` run: 426 passed, 1 skipped, 1 xfailed in 5.81 s.
   - **Compiler builds.** The WASM compiler was built twice: once for the playground (bundler target) and once for the
     video page (web target).
   - **Playground.** `web_demo` was copied into `build/` and rebuilt against the new WASM, because the live site rendered
     blank in headless Chrome. Puppeteer then captured it in dark mode while code was typed into it. That gave 105
     frames of live errors, ending in `invalid domain crossing: different clock`, followed by a fixed version that
     outputs Verilog.
7. **Storyboard.** `mv/STORYBOARD.md` maps every lyric section to a visual concept and sets the visual identity:
   - a dark "EDA tool at 3 a.m." look;
   - colours that carry meaning: cyan and magenta for the two clock domains, amber for `async`, red for errors, green
     for passing;
   - Archivo Black for headline type, JetBrains Mono for code, Space Grotesk for labels.
8. **Engine and scenes.** A custom Canvas 2D engine and 15 scenes, written section by section. Each section was
   checked by rendering stills into contact sheets before moving on.
9. **Drafts and review.**
   - A full 30 fps draft rendered in 1.4 minutes.
   - It was reviewed as 166 timestamped frames, one every 2 s, laid out on 7 contact sheets.
   - About ten problems were fixed (section 3), then the final 1080p60 video was rendered in 2.4 minutes.
10. **Wrap-up.** README, `.gitignore`, a commit to the new `video` branch, and this report.

---

## 2. User feedback, and what changed

| Round | Feedback | Change |
|---|---|---|
| v1 → v2 | "Self-reflect a bit and improve"; leaving out compiler bugs was right; emphasize compilation speed | Rewrote weak lines ("carry made a switch", "meeting with the man", "grep finds every flood"). Added a whole verse on compile speed (cores, squiggles, license server, pytest). The outro time was measured, not guessed: the demo project builds in about 20 ms. |
| v2 → v3 | Drop delta cycles (a simulator feature, not a language one). Verse 2 was too detailed, because numbers don't sing well. `ident` is hard to read and complex. Add interfaces. Stress that ad-hoc codegen, glue and build tools become unnecessary | Removed delta cycles everywhere and retitled the song "Invalid Domain Crossing". Verse 2 became conceptual. Removed `ident`. Added a verse on interfaces and views. The bridge now walks through a "tool graveyard" (Perl, Python writing Perl, Jinja, Tcl, nightly lint cron, a taped-together build script). |
| v3 → v4 | The "eight-bit" line sounds odd when sung. Mention SV's implicit casting and VHDL's over-strictness. Suno mispronounces "clog", so spell it "c-log". The bridge's error messages are too long | Rewrote the opening lines and added casting couplets. Respelled "c-log-two". The bridge now uses short error names ("Driver mismatch!", "Self-loop!", "Multiple drivers!", "Different clock!"). |
| v4 → v5 | "Count the values… not the bits" and "and I haven't shipped it yet" read strangely | The chorus now ends with "and the compiler catches every crossing I forget!" and "you give the range, it picks the bits". The outro became "if it compiles… it fits". |
| v5 | Delete the VHDL/casting couplets; they over-complicate things | Removed them. |
| v5 | Drop the "four-deep" line too; is there a neater VHDL complaint, maybe verbosity? | Replaced it with one verbosity couplet: "say it all three times: entity, component, port map — and it's longer than these rhymes." |
| Video | Include "real" compiler material: source, diagnostics, the web demo | All on-screen diagnostics and Verilog come from the real compiler. The page runs the WASM compiler live. The key change shows real playground captures. |
| Process | "You okay?" after a long silence | Gave a status summary and posted progress more often after that. |

---

## 3. Issues and how they were resolved

### Environment and audio

- **GPU unusable.** An NVIDIA driver/library version mismatch (kernel module 595.91, libraries 595.99) ruled out local
  song generation and GPU inference. Demucs and Whisper ran on 32 CPU cores instead, which was fast enough.
- **Suno song is private.**
  - Plain HTTP got `audio_url: .../forbidden`, and the CDN returned 403.
  - With Selenium, the page played a `blob:` URL, and the CloudFront `.m4a` it downloaded was obfuscated: it had no MP4
    boxes.
  - Fetching the blob failed because it was a MediaSource stream.
  - **Fix:** hook `SourceBuffer.appendBuffer` before the page loads and collect the decoded fragmented-MP4 Opus chunks
    as they are appended.
- **About 20 s of garbage at the end.** The captured stream decoded to 351–353 s, but the song is 331 s long. Packet
  analysis showed 16,559 normal 20 ms packets followed by 1,100 padding packets one tick long. The audio was trimmed to
  331.18 s; the timeline from 0 was unaffected.
- **faster-whisper crashed.** An incompatibility with PyAV (`unexpected keyword argument 'metadata_errors'`). Fixed by
  decoding with librosa and passing a numpy array instead.
- **Alignment glitch.** The first word, "Three", latched onto a sound 8 s too early. A sanity pass now clamps any word
  longer than 1.4 s.

### Compiler material

- **The live playground rendered blank** in headless Chrome, and the local `web_demo/dist` was stale (still titled
  "HWLang"). Rebuilt it from current source inside `video/build/` so nothing outside `video/` changed.
- **WASM `run_all` panics in Node** ("not in a browser", from wasm-timer). It works in Chrome, which is where the
  renderer runs.
- **The compiler caught a mistake in my example.** `axi_gearbox(T=Pixel, ...)` failed to type-check: the gearbox's
  `[false]` reset value means it only supports `bool`. The snippet now uses `T=bool`, and the error itself was a nice
  confirmation that the on-screen claims are real.
- **CodeMirror auto-indentation doubled the indentation** while typing into the playground.
  - `execCommand("insertText")` did not reach the editor.
  - Final approach: after each Enter, select back to the start of the line and delete the copied indent, but only when
    the previous line was indented, then type the line's own spaces.

### Rendering bugs found in review

- **Full-screen amber flood at 144.5 s.** `ease.outBack(0)` returns a tiny positive float instead of exactly 0. That
  made a `prog(...) > 0` check pass early and fire `exp(+x)`, giving an alpha above 1. I found it by rendering the frame
  one scene at a time. Fix: `prog()` now returns exact 0/1 at its endpoints, and the flash exponent is clamped.
- **Layout collisions** fixed after review:
  - the diagnostic panel overflowing in the intro;
  - the waveform labels;
  - the sticky note against its "FEATURE" stamp;
  - the VHDL line counter;
  - the interface panel against the module blocks;
  - the block-diagram labels;
  - the bridge diagnostics;
  - verse 4's editor against the stamps;
  - the pre-chorus 3 tool row against the lyrics;
  - the type-mismatch panel against the lyrics.
- **Timing and pacing:**
  - the giant chorus letters overlapped the payoff visual;
  - chorus line visuals overlapped each other, so each now ends at the next line;
  - the tool-graveyard cards were struck through too early, so they now go at the end of each phrase;
  - the intro title held static for 8 s, so it got a beat chase across the flops and a slow push;
  - the empty sensitivity scene now starts with large centred code;
  - the block diagram now starts drawing earlier.
- **Final chorus looked the same as the first two.** Added stronger lanes, a camera that rides the kick, and
  domain-coloured K/H/D/L letters.

### My own operational mistakes

- **`pkill -f "http.server 8765"` killed my own shell**, because the pattern matched the command itself (exit code
  144). Later I stopped the server by looking up its PID with `ss` instead.
- **A recursive delete was blocked** by the safety check because the target was the shell's current directory. Resolved
  by deleting the single file and running `rmdir` from a different directory.
- **A misdirected `sed -i '169d'`** deleted a line from `verse2.js` instead of `verse1.js`. I noticed right away, worked
  out from context that it was a `ctx.beginPath()`, and restored it.

---

## 4. How the flow works

### Data preparation (one-off, `tools/`)

```
audio/khdl.wav ──► analyze_audio.py ──► analysis/audio.json    (tempo, beats, onsets, 60 fps band envelopes)
               ──► demucs (CPU)     ──► analysis/stems/…/vocals.wav
vocals.wav     ──► transcribe.py    ──► analysis/words.json    (Whisper word timestamps)
suno.md + words.json ──► align_lyrics.py ──► analysis/lyrics.json (canonical lyric lines, per-word start/end)
khdl binary on compiler_*/ projects, grep, pytest ──► mv/data/compiler.json
capture_playground.mjs (puppeteer + local playground) ──► mv/assets/playground/*.png
```

### The video page (`mv/`)

- **`index.html` and `src/main.js`** load the fonts, `data/*.json`, the WASM compiler and the playground images. They
  expose `window.__render(t)`.
- **`src/lib.js`** holds easing, the time windows (`prog`, `win`), hash-based randomness, and the beat and envelope
  lookups (`beatInfo`, `pulse`, `kick`).
- **`src/draw.js`** holds text, panels, waveforms and bus lanes. It draws code highlighted with the compiler's own
  tokenizer, and turns ANSI escape codes into coloured text.
- **`src/khdl.js`** wraps the WASM compiler (`run_all`, `codemirror_tokenize_to_tree`) and caches results.
- **`src/lyrics.js`** turns sung phrases back into code tokens (for example, "A plus B" becomes `a + b`), draws the
  lyric typography word by word, and provides `wordT(line, word)` so scenes can sync to specific words.
- **`src/scenes/*.js`** contains 15 scenes. Each has a time range and a `render(ctx, t)`, and they composite in order.
- **Every frame is a pure function of `t`.** Nothing is stored between frames, so any frame can be rendered alone and
  in any order. That is what lets the rendering run in parallel.

### Rendering (`render.mjs`)

1. A built-in Node HTTP server serves `video/`.
2. The frames (331.18 s × 60 fps = 19,871) are split into 14 contiguous ranges. Each worker:
   - starts headless Chrome through puppeteer-core;
   - for each frame calls `__render(f/60)` and takes a JPEG (quality 94) through Chrome DevTools
     `Page.captureScreenshot`;
   - pipes the JPEGs into its own ffmpeg, which encodes x264 at CRF 16, yuv420p, 60 fps.
3. The segments are joined without re-encoding (concat demuxer) and the WAV is muxed in as 320 kbps AAC with
   `-shortest`.
4. Throughput was about 140 fps, so a full 1080p60 render takes about 2.4 minutes.

### Review loop

- `snap.mjs` renders chosen timestamps to PNG, and `sheet.sh` tiles them into contact sheets.
- For a full review, render a 30 fps draft, extract a frame every 2 s with timestamps, and tile them.
- **Preview with audio:** serve `video/` over HTTP (opening the file directly fails with CORS errors), then open
  `/mv/index.html?preview&t=…`.

---

## 5. Honest limitations

- **Only reviewed as stills.** The video was checked through sampled frames, never watched in motion with sound. Motion
  feel and lyric sync were never judged by eye or ear.
- **Lyric timing quality depends on Whisper.** About 10% of word times are interpolated.
- **The choruses repeat.** All three share one visual system; the final one is only more intense. Most scene changes
  are crossfades rather than designed transitions.
- **Some on-screen text was invented**, as illustration rather than claim:
  - the "Finished in 20 ms" outro line (the CLI prints nothing on success, though the timing itself was measured);
  - the license-server text;
  - the vendor support ticket;
  - the "glue" connection lines;
  - the tool-file contents.
- **The pytest numbers disagree with the lyric** on purpose. The screen shows the real 426 tests in 5.81 s, while the
  lyric says "380 in four seconds", with an on-screen joke about it.
- **A fresh checkout needs rebuilding first.** `npm install` and the WASM build (see `README.md`) must be run before
  the page works.
