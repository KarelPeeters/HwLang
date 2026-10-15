# "Invalid Domain Crossing": a KHDL music video

**Output:** `out/khdl_invalid_domain_crossing.mp4` (1920×1080, 60 fps, 5:31)

## Contents

| Path | What |
|---|---|
| `research/khdl_summary.md` | Background summary of the language, used to write the lyrics |
| `lyrics_v*.md`, `suno.md` | Lyric drafts. `suno.md` is the final text as it was sung |
| `audio/khdl.wav` | The Suno track, trimmed to its real length |
| `analysis/` | Beat and envelope analysis (`audio.json`), demucs stems, Whisper words, aligned lyrics |
| `tools/` | `analyze_audio.py`, `transcribe.py`, `align_lyrics.py` |
| `compiler_demo/`, `compiler_ok/`, `compiler_sync/`, `compiler_bridge/` | Small projects compiled with the real `khdl` binary; the diagnostics and Verilog in the video come from these |
| `build/` | WASM build of the compiler and a local copy of the playground (`web_demo`) |
| `mv/` | The video itself: a Canvas 2D page where every frame is a pure function of time |

## The video page (`mv/`)

- **Scenes:** `src/scenes/*.js`, timed against the lyric words (`wordT("line", "word")`).
- **Real compiler in the page:** `src/khdl.js` loads the real compiler as WebAssembly (`vendor/khdl_wasm`).
  - Code highlighting uses the compiler's own tokenizer.
  - The editor squiggles in verse 4 come from `run_all()` running live.
- **Playground frames:** `assets/playground/` holds frames captured from the real playground by `capture_playground.mjs`.

## Commands

All run from `mv/`.

- **Preview with audio:** serve `video/` (for example `python3 -m http.server` in `video/`), then open
  `/mv/index.html?preview&t=82`. Space plays and pauses; the arrow keys seek.
- **Review frames:** `node snap.mjs ../shots/x 12.5 97.3`
- **Render:** `node render.mjs --fps=60 --workers=14 --out=../out/khdl_invalid_domain_crossing.mp4`. This takes
  about 2.5 minutes.

## Regenerating ignored files

- **Dependencies:** `npm install` in `mv/` and in `build/web_demo/`.
- **WASM compiler for the page:** run this in `rust/khdl_wasm`:
  `wasm-pack build --release --target web --out-dir ../../video/mv/vendor/khdl_wasm`
- **Audio:** put the track at `audio/khdl.wav`, trimmed to 331.18 s. Then rerun the `tools/` scripts to recreate
  `analysis/stems`.
