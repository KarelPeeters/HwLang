"""Align canonical lyrics (from suno.md) to Whisper word timestamps.

Output: list of lines with per-word start/end times and the section they belong to.
"""

import difflib
import json
import re
import sys

suno_path, words_path, out_path = sys.argv[1:4]

text = open(suno_path).read()
lyr = text.split("## Lyrics", 1)[1].split("## Tips", 1)[0]

lines = []
section = None
for raw in lyr.splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("(paste"):
        continue
    m = re.fullmatch(r"\[(.+)\]", raw)
    if m:
        tag = m.group(1)
        if not (tag.startswith("Spoken") or tag.startswith("Rapid") or tag.startswith("Half-time")):
            section = tag
        continue
    lines.append({"section": section, "text": raw})


def norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9']", "", w)
    return w


canon = []  # (line index, word index, normalized)
for li, line in enumerate(lines):
    line["words"] = []
    for wi, w in enumerate(line["text"].split()):
        line["words"].append({"text": w})
        n = norm(w)
        canon.append((li, wi, n))

heard = json.load(open(words_path))
hn = [norm(w["w"]) for w in heard]

# split hyphenated / letter-spelled tokens on the whisper side a bit
sm = difflib.SequenceMatcher(a=[c[2] for c in canon], b=hn, autojunk=False)
times = [None] * len(canon)
for a, b, size in sm.get_matching_blocks():
    for k in range(size):
        h = heard[b + k]
        times[a + k] = (h["s"], h["e"])

# fuzzy-match leftover canon words against nearby heard words (similar spelling)
matched_b = {}
for a, b, size in sm.get_matching_blocks():
    for k in range(size):
        matched_b[a + k] = b + k
for i in range(len(canon)):
    if times[i] is not None:
        continue
    prev = max([j for j in range(i) if j in matched_b], default=None)
    nxt = min([j for j in range(i + 1, len(canon)) if j in matched_b], default=None)
    lo = matched_b[prev] + 1 if prev is not None else 0
    hi = matched_b[nxt] if nxt is not None else len(heard)
    best, best_r = None, 0.6
    for b in range(lo, hi):
        r = difflib.SequenceMatcher(a=canon[i][2], b=hn[b]).ratio()
        if r > best_r:
            best, best_r = b, r
    if best is not None:
        times[i] = (heard[best]["s"], heard[best]["e"])
        matched_b[i] = best

# interpolate remaining gaps linearly between known neighbours
known = [i for i, t in enumerate(times) if t is not None]
for i in range(len(canon)):
    if times[i] is not None:
        continue
    prev = max([j for j in known if j < i], default=None)
    nxt = min([j for j in known if j > i], default=None)
    if prev is None:
        s = times[nxt][0] - 0.3 * (nxt - i)
        times[i] = (s, s + 0.25)
    elif nxt is None:
        s = times[prev][1] + 0.3 * (i - prev)
        times[i] = (s, s + 0.25)
    else:
        t0, t1 = times[prev][1], times[nxt][0]
        f0 = (i - prev) / (nxt - prev)
        f1 = (i - prev + 1) / (nxt - prev)
        times[i] = (t0 + (t1 - t0) * f0 * 0.98, t0 + (t1 - t0) * f1 * 0.98)

for (li, wi, _), (s, e) in zip(canon, times):
    lines[li]["words"][wi]["s"] = round(s, 3)
    lines[li]["words"][wi]["e"] = round(e, 3)

for line in lines:
    ws = line["words"]
    for i, w in enumerate(ws):
        # a single sung word never lasts this long; whisper latched onto an earlier sound
        if w["e"] - w["s"] > 1.4:
            w["s"] = round(w["e"] - 0.45, 3)
        if i + 1 < len(ws) and ws[i + 1]["s"] - w["e"] > 1.4:
            w["s"] = round(ws[i + 1]["s"] - 0.5, 3)
            w["e"] = round(ws[i + 1]["s"] - 0.05, 3)

for line in lines:
    line["s"] = line["words"][0]["s"]
    line["e"] = line["words"][-1]["e"]

n_direct = sum(1 for a, b, size in sm.get_matching_blocks() for _ in range(size))
print(f"canonical words {len(canon)}, heard {len(heard)}, exact matches {n_direct}")
for line in lines:
    print(f"{line['s']:7.2f} {line['e']:7.2f}  [{line['section']}] {line['text']}")
json.dump(lines, open(out_path, "w"), indent=1)
