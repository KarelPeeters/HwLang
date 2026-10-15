import json
import sys

import librosa
from faster_whisper import WhisperModel

path, out = sys.argv[1], sys.argv[2]
model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=32)
audio, _ = librosa.load(path, sr=16000, mono=True)
segments, info = model.transcribe(
    audio,
    language="en",
    word_timestamps=True,
    vad_filter=False,
    beam_size=5,
    condition_on_previous_text=False,
    initial_prompt="KHDL, Verilog, VHDL, sensitivity list, delta, latch, netlist, clocked, comb, reg wire, undef, interface, "
    "async to sync, invalid domain crossing, c-log-two, pytest, Verilator, Jinja, Tcl, grep.",
)
words = []
for seg in segments:
    print(f"[{seg.start:7.2f} -> {seg.end:7.2f}] {seg.text}", flush=True)
    for w in seg.words:
        words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)})
json.dump(words, open(out, "w"), indent=0)
