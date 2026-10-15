// Lyric layer: word-timed kinetic typography, with sung phrases mapped back to code.

import { C, W, H, clamp, ease, hexA, prog } from "./lib.js";
import { F, font } from "./draw.js";

const norm = (w) => w.toLowerCase().replace(/[^a-z0-9']/g, "");

// sung words -> displayed token
const RULES = [
  { m: ["three", "am"], text: "3 a.m." },
  { m: ["a", "plus", "b"], text: "a + b", code: true },
  { m: ["contextdetermined", "width"], text: "“context-determined width,”", color: C.amber },
  { m: ["else"], text: "else", code: true },
  { m: ["case"], text: "case", code: true },
  { m: ["entity"], text: "entity,", code: true },
  { m: ["component"], text: "component,", code: true },
  { m: ["port", "map"], text: "port map,", code: true },
  { m: ["fortyone"], text: "41" },
  { m: ["clogtwo"], text: "$clog2", code: true },
  { m: ["type", "mismatch"], text: "type mismatch,", code: true, color: C.red },
  { m: ["oneflop"], text: "one-flop" },
  { m: ["comb"], text: "comb", code: true },
  { m: ["clocked"], text: "clocked,", code: true },
  { m: ["reg", "wire"], text: "reg wire", code: true },
  { m: ["equals", "undef"], text: "= undef", code: true },
  { m: ["dot", "valid", "valid"], text: ".valid(valid),", code: true },
  { m: ["dot", "ready", "ready"], text: ".ready(ready),", code: true },
  { m: ["interface"], text: "interface", code: true },
  { m: ["view"], text: "view:", code: true },
  { m: ["input"], text: "input", code: true },
  { m: ["output"], text: "output", code: true },
  { m: ["driver", "mismatch"], text: "driver mismatch!", code: true, color: C.red },
  { m: ["selfloop"], text: "self-loop!", code: true, color: C.red },
  { m: ["multiple", "drivers"], text: "multiple drivers!", code: true, color: C.red },
  { m: ["different", "clock"], text: "different clock!", code: true, color: C.red },
  { m: ["perl"], text: "Perl", code: true },
  { m: ["jinja"], text: "Jinja", code: true },
  { m: ["tcl"], text: "Tcl", code: true },
  { m: ["unsafe", "value", "with", "domain"], text: "unsafe_value_with_domain", code: true, color: C.red },
  { m: ["grep"], text: "grep", code: true },
  { m: ["pytest"], text: "pytest", code: true },
  { m: ["three", "hundred", "eighty"], text: "380" },
  { m: ["four", "seconds"], text: "4 seconds," },
  { m: ["verilog", "twothousandfive"], text: "Verilog 2005," },
  { m: ["k", "h", "d", "l", "build"], text: "$ khdl build:", code: true },
  { m: ["twenty", "milliseconds"], text: "20 ms." },
  { m: ["async"], text: "async", code: true, color: C.amber },
  { m: ["sync"], text: "sync.", code: true, color: C.cyan },
];

let LINES = [];

export function setLyrics(raw) {
  LINES = raw.map((line, li) => {
    const ws = line.words;
    const toks = [];
    for (let i = 0; i < ws.length; ) {
      let hit = null;
      for (const r of RULES) {
        if (i + r.m.length > ws.length) continue;
        if (r.m.every((m, k) => norm(ws[i + k].text) === m)) {
          hit = r;
          break;
        }
      }
      if (hit) {
        const last = ws[i + hit.m.length - 1];
        // keep trailing punctuation from the sung text when the rule text has none
        let txt = hit.text;
        const tail = last.text.match(/[.,:;!?…]+$/);
        if (tail && !/[.,:;!?…”]$/.test(txt)) txt += tail[0];
        toks.push({ text: txt, s: ws[i].s, e: last.e, code: !!hit.code, color: hit.color });
        i += hit.m.length;
      } else {
        toks.push({ text: ws[i].text.replace(/"/g, ""), s: ws[i].s, e: ws[i].e });
        i++;
      }
    }
    return { ...line, idx: li, toks };
  });
}

export function lines() {
  return LINES;
}

// lines visible at time t (current line, with fade timing)
export function activeLine(t, lead = 0.25) {
  let best = null;
  for (let i = 0; i < LINES.length; i++) {
    const l = LINES[i];
    const next = LINES[i + 1];
    const end = Math.min(next ? next.s - 0.08 : 1e9, l.e + 1.6);
    if (t >= l.s - lead && t < end) best = { line: l, end, next };
  }
  return best;
}

export function lineAt(idx) {
  return LINES[idx];
}

export function findLine(sub, after = 0) {
  return LINES.find((l) => l.s >= after && l.text.toLowerCase().includes(sub.toLowerCase()));
}

// layout tokens into wrapped rows
function layout(ctx, toks, size, maxW, fam, weight) {
  const rows = [[]];
  let x = 0;
  const space = size * 0.28;
  for (const tk of toks) {
    font(ctx, tk.code ? size * 0.9 : size, tk.code ? F.mono : fam, tk.code ? 700 : weight);
    const w = ctx.measureText(tk.text).width;
    if (x > 0 && x + w > maxW) {
      rows.push([]);
      x = 0;
    }
    rows[rows.length - 1].push({ tk, x, w });
    x += w + space;
  }
  return rows.map((r) => ({ items: r, w: r.length ? r[r.length - 1].x + r[r.length - 1].w : 0 }));
}

// style: "lower" (lower third) | "center" (big) | "top"
export function drawLyrics(ctx, t, { style = "lower", alpha = 1, size, color = C.text, maxW, y, x, align } = {}) {
  const a = activeLine(t);
  if (!a || alpha <= 0) return;
  const { line, end } = a;
  const big = style === "center";
  size = size ?? (big ? 92 : 50);
  const fam = big ? F.disp : F.sans;
  const weight = big ? 400 : 600;
  maxW = maxW ?? (big ? 1500 : 1500);
  const rows = layout(ctx, line.toks, size, maxW, fam, weight);
  const lh = size * 1.18;
  align = align ?? (big ? "center" : "left");
  const baseY = y ?? (big ? H / 2 - ((rows.length - 1) * lh) / 2 + size * 0.35 : H - 110 - (rows.length - 1) * lh);
  const out = clamp((end - t) / 0.18);
  ctx.save();
  ctx.globalAlpha = alpha * out;
  if (!big) {
    // backing scrim for legibility
    const sw = Math.max(...rows.map((r) => r.w)) + 60;
    const sx = (x ?? 120) - 30;
    const g = ctx.createLinearGradient(0, baseY - size - 30, 0, baseY + (rows.length - 1) * lh + 40);
    g.addColorStop(0, "rgba(10,12,17,0)");
    g.addColorStop(0.35, "rgba(10,12,17,0.72)");
    g.addColorStop(1, "rgba(10,12,17,0.72)");
    ctx.fillStyle = g;
    ctx.fillRect(sx, baseY - size - 30, sw, (rows.length - 1) * lh + size + 70);
    ctx.fillStyle = C.cyan;
    ctx.fillRect(sx, baseY - size - 4, 4, (rows.length - 1) * lh + size + 22);
  }
  rows.forEach((row, ri) => {
    const rx = align === "center" ? W / 2 - row.w / 2 : x ?? 120;
    const ry = baseY + ri * lh;
    for (const { tk, x: ox } of row.items) {
      const p = clamp((t - tk.s + 0.06) / 0.16);
      if (p <= 0) continue;
      const active = t >= tk.s - 0.05 && t < tk.e + 0.15;
      const e = ease.outBack(p, 2.2);
      const dy = (1 - e) * size * 0.35;
      const tsize = tk.code ? size * 0.9 : size;
      font(ctx, tsize, tk.code ? F.mono : fam, tk.code ? 700 : weight);
      ctx.globalAlpha = alpha * out * clamp(p * 1.4);
      const col = tk.color || (tk.code ? C.cyan : color);
      if (tk.code) {
        const w = ctx.measureText(tk.text).width;
        ctx.fillStyle = hexA(col, 0.12);
        ctx.fillRect(rx + ox - 6, ry + dy - tsize * 0.86, w + 12, tsize * 1.12);
      }
      ctx.fillStyle = col;
      if (active && big) {
        ctx.shadowColor = col;
        ctx.shadowBlur = 24;
      } else ctx.shadowBlur = 0;
      ctx.fillText(tk.text, rx + ox, ry + dy);
    }
  });
  ctx.restore();
}

// time at which `word` is sung in the first line containing `lineSub` (searching from `after`)
export function wordT(lineSub, word, after = 0, which = "s") {
  const l = findLine(lineSub, after);
  if (!l) throw new Error(`no line ${lineSub}`);
  if (!word) return l[which];
  const w = l.words.find((w) => norm(w.text).startsWith(norm(word)));
  if (!w) throw new Error(`no word ${word} in ${lineSub}`);
  return w[which];
}

// token-level helper for scenes that typeset lyrics themselves
export function tokAlpha(tk, t) {
  return prog(t, tk.s - 0.06, 0.16);
}
