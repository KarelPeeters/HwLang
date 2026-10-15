// Drawing primitives: text, code (highlighted with the real KHDL tokenizer), ANSI diagnostics, panels, waveforms.

import { C, W, H, clamp, hexA, rnd } from "./lib.js";
import { khdlTokens } from "./khdl.js";

export const F = {
  mono: "JetBrains Mono",
  sans: "Space Grotesk",
  disp: "Archivo Black",
};

export function font(ctx, size, fam = F.sans, weight = 500) {
  ctx.font = `${weight} ${size}px "${fam}"`;
}

export function text(ctx, s, x, y, { size = 40, fam = F.sans, weight = 500, color = C.text, align = "left", base = "alphabetic", alpha = 1, spacing = 0 } = {}) {
  if (alpha <= 0) return 0;
  font(ctx, size, fam, weight);
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = base;
  if (spacing) ctx.letterSpacing = `${spacing}px`;
  ctx.fillText(s, x, y);
  const w = ctx.measureText(s).width;
  ctx.restore();
  return w;
}

export function measure(ctx, s, size, fam = F.sans, weight = 500, spacing = 0) {
  font(ctx, size, fam, weight);
  ctx.save();
  if (spacing) ctx.letterSpacing = `${spacing}px`;
  const w = ctx.measureText(s).width;
  ctx.restore();
  return w;
}

export function glow(ctx, color, blur, fn) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  fn();
  ctx.restore();
}

export function rect(ctx, x, y, w, h, color, alpha = 1) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}

export function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

export function strokeRect(ctx, x, y, w, h, color, lw = 1, alpha = 1) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  ctx.restore();
}

export function line(ctx, x1, y1, x2, y2, color, lw = 1, alpha = 1) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

// window/panel chrome
export function panel(ctx, x, y, w, h, { title = "", alpha = 1, accent = C.dim, fill = C.panel, radius = 10 } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.shadowColor = "rgba(0,0,0,0.55)";
  ctx.shadowBlur = 40;
  ctx.shadowOffsetY = 18;
  ctx.fillStyle = fill;
  rrect(ctx, x, y, w, h, radius);
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.strokeStyle = C.rule;
  ctx.lineWidth = 1;
  ctx.stroke();
  if (title) {
    ctx.fillStyle = "#0e1118";
    ctx.beginPath();
    ctx.roundRect(x, y, w, 36, [radius, radius, 0, 0]);
    ctx.fill();
    line(ctx, x, y + 36, x + w, y + 36, C.rule);
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = [C.red, C.amber, C.green][i];
      ctx.globalAlpha = 0.75 * alpha;
      ctx.beginPath();
      ctx.arc(x + 20 + i * 18, y + 18, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = alpha;
    text(ctx, title, x + w / 2, y + 24, { size: 15, fam: F.mono, weight: 400, color: C.muted, align: "center" });
    ctx.fillStyle = accent;
  }
  ctx.restore();
}

// ---------- code rendering ----------

const KH_DOMAIN = new Set(["async", "sync", "clock", "clocked", "comb", "reg", "wire", "undef"]);
const KH_TYPES = new Set(["bool", "int", "uint", "type", "natural", "Option", "any"]);
export const CODE = {
  keyword: "#c49bff",
  domain: C.cyan,
  type: C.amber,
  number: "#7ef0b8",
  string: "#7ef0b8",
  name: "#dfe6f2",
  call: "#8fc6ff",
  punctuation: "#7c8597",
  comment: "#4f5768",
  unsafe: C.red,
};

const tokCache = new Map();

// returns array of {s, e, kind} covering source (gaps are whitespace)
export function khdlSpans(src) {
  if (tokCache.has(src)) return tokCache.get(src);
  const toks = khdlTokens(src);
  const out = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    const word = src.slice(t.s, t.e);
    let kind = t.kind;
    if (kind === "keyword" && KH_DOMAIN.has(word)) kind = "domain";
    else if (kind === "name") {
      if (KH_TYPES.has(word)) kind = "type";
      else if (word.startsWith("unsafe_")) kind = "unsafe";
      else if (toks[i + 1] && src.slice(toks[i + 1].s, toks[i + 1].e) === "(") kind = "call";
    }
    out.push({ s: t.s, e: t.e, kind });
  }
  tokCache.set(src, out);
  return out;
}

const VLOG_KW = new Set(("module endmodule input output reg wire always begin end if else assign posedge negedge case endcase default " +
  "localparam parameter integer genvar generate endgenerate for signed unsigned logic always_ff always_comb entity is port architecture of " +
  "signal component map std_logic std_logic_vector downto in out begin process when others then elsif end").split(" "));

export function vlogSpans(src) {
  if (tokCache.has("v:" + src)) return tokCache.get("v:" + src);
  const out = [];
  const re = /(\/\/[^\n]*|--[^\n]*)|("[^"\n]*")|(\$?[A-Za-z_][A-Za-z0-9_]*)|(\d+'[bdhsBDHS]?[0-9a-fA-FxXzZ_]+|\d+)|([^\sA-Za-z0-9_])/g;
  let m;
  while ((m = re.exec(src))) {
    let kind = m[1] ? "comment" : m[2] ? "string" : m[3] ? (VLOG_KW.has(m[3].toLowerCase()) ? "keyword" : m[3].startsWith("$") ? "call" : "name") : m[4] ? "number" : "punctuation";
    out.push({ s: m.index, e: m.index + m[0].length, kind });
  }
  tokCache.set("v:" + src, out);
  return out;
}

// Draw source code. opts: size, lh, chars (reveal count), lang, alpha, lineAlpha(fn), colorOverride(fn(span)->color)
export function drawCode(ctx, src, x, y, opts = {}) {
  const { size = 24, lh = size * 1.45, chars = Infinity, lang = "khdl", alpha = 1, cursor = false, lineAlpha = null, tint = null, maxLines = Infinity } = opts;
  if (alpha <= 0) return { lines: 0, cw: 0 };
  const spans = lang === "khdl" ? khdlSpans(src) : vlogSpans(src);
  font(ctx, size, F.mono, 400);
  const cw = ctx.measureText("M").width;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.textBaseline = "alphabetic";
  // map char index -> (line, col)
  const lineStarts = [0];
  for (let i = 0; i < src.length; i++) if (src[i] === "\n") lineStarts.push(i + 1);
  const pos = (i) => {
    let lo = 0;
    let hi = lineStarts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (lineStarts[mid] <= i) lo = mid;
      else hi = mid - 1;
    }
    return [lo, i - lineStarts[lo]];
  };
  for (const sp of spans) {
    if (sp.s >= chars) break;
    const e = Math.min(sp.e, chars);
    const [ln, col] = pos(sp.s);
    if (ln >= maxLines) break;
    const la = lineAlpha ? lineAlpha(ln) : 1;
    if (la <= 0) continue;
    ctx.globalAlpha = alpha * la;
    ctx.fillStyle = (tint && tint(sp, src.slice(sp.s, sp.e), ln)) || CODE[sp.kind] || CODE.name;
    ctx.fillText(src.slice(sp.s, e), x + col * cw, y + ln * lh);
  }
  if (cursor && chars < Infinity) {
    const [ln, col] = pos(Math.min(chars, src.length));
    ctx.globalAlpha = alpha;
    ctx.fillStyle = C.cyan;
    ctx.fillRect(x + col * cw, y + ln * lh - size * 0.85, Math.max(2, size * 0.1), size * 1.05);
  }
  ctx.restore();
  return { lines: lineStarts.length, cw, lh, pos };
}

// ---------- ANSI diagnostics ----------
const ANSI_COLORS = { 91: C.red, 94: "#5aa9ff", 92: C.green, 93: C.amber, 96: C.cyan, 95: C.magenta };

export function parseAnsi(s) {
  const lines = [];
  let cur = [];
  let color = null;
  let bold = false;
  const re = /\x1b\[([0-9;]*)m|([^\x1b\n]+)|(\n)/g;
  let m;
  while ((m = re.exec(s))) {
    if (m[1] !== undefined) {
      for (const code of m[1].split(";").map(Number)) {
        if (code === 0) {
          color = null;
          bold = false;
        } else if (code === 1) bold = true;
        else if (ANSI_COLORS[code]) color = ANSI_COLORS[code];
      }
    } else if (m[2] !== undefined) cur.push({ text: m[2], color, bold });
    else {
      lines.push(cur);
      cur = [];
    }
  }
  lines.push(cur);
  return lines;
}

const ansiCache = new Map();
export function drawAnsi(ctx, s, x, y, { size = 20, lh = size * 1.45, alpha = 1, lines: maxLines = Infinity, chars = Infinity } = {}) {
  if (alpha <= 0) return 0;
  let parsed = ansiCache.get(s);
  if (!parsed) ansiCache.set(s, (parsed = parseAnsi(s)));
  ctx.save();
  ctx.globalAlpha *= alpha;
  let n = 0;
  for (let li = 0; li < parsed.length && li < maxLines; li++) {
    let cx = x;
    for (const run of parsed[li]) {
      font(ctx, size, F.mono, run.bold ? 700 : 400);
      let t = run.text;
      if (n + t.length > chars) t = t.slice(0, Math.max(0, chars - n));
      n += run.text.length;
      ctx.fillStyle = run.color || "#c9d1de";
      ctx.fillText(t, cx, y + li * lh);
      cx += ctx.measureText(run.text).width;
    }
    n += 1;
  }
  ctx.restore();
  return parsed.length;
}

export function ansiPlain(s) {
  return s.replace(/\x1b\[[0-9;]*m/g, "");
}

// ---------- background ----------
export function dotGrid(ctx, t, { alpha = 1, spacing = 40, ox = 0, oy = 0, color = C.grid } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  const sx = ((ox % spacing) + spacing) % spacing;
  const sy = ((oy % spacing) + spacing) % spacing;
  for (let y = sy; y < H; y += spacing) for (let x = sx; x < W; x += spacing) ctx.fillRect(x - 1, y - 1, 2, 2);
  ctx.restore();
}

export function vignette(ctx, strength = 0.6) {
  const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H * 0.95);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, `rgba(0,0,0,${strength})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

export function grain(ctx, t, amount = 0.05) {
  // cheap deterministic film grain: sparse random dots
  const f = Math.floor(t * 60);
  ctx.save();
  for (let i = 0; i < 1400; i++) {
    const x = rnd(i, f, 1) * W;
    const y = rnd(i, f, 2) * H;
    const v = rnd(i, f, 3);
    ctx.fillStyle = v > 0.5 ? `rgba(255,255,255,${amount})` : `rgba(0,0,0,${amount * 1.6})`;
    ctx.fillRect(x, y, 2, 2);
  }
  ctx.restore();
}

// ---------- waveforms ----------
// bits: function(i) -> 0/1 for cycle i; draws a digital waveform lane
export function wave(ctx, x, y, w, h, { period = 60, phase = 0, bits = null, color = C.cyan, lw = 2.5, reveal = 1, alpha = 1, glowAmt = 12 } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.beginPath();
  ctx.rect(x, y - h - 10, w * clamp(reveal), h + 30);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.lineJoin = "miter";
  ctx.shadowColor = color;
  ctx.shadowBlur = glowAmt;
  ctx.beginPath();
  const half = period / 2;
  const start = Math.floor(-phase / half) - 1;
  let prev = null;
  for (let k = start; ; k++) {
    const sx = x + k * half + phase;
    if (sx > x + w) break;
    const v = bits ? bits(k) : k & 1 ? 0 : 1;
    const yy = v ? y - h : y;
    if (prev === null) ctx.moveTo(sx, yy);
    else {
      ctx.lineTo(sx, prev);
      ctx.lineTo(sx, yy);
    }
    prev = yy;
    ctx.lineTo(sx + half, yy);
  }
  ctx.stroke();
  ctx.restore();
}

// bus lane with hex values (GTKWave-like)
export function busLane(ctx, x, y, w, h, cells, { color = C.green, alpha = 1, size = 18, reveal = 1 } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.beginPath();
  ctx.rect(x, y - 4, w * clamp(reveal), h + 8);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (const c of cells) {
    const x0 = x + c.x0;
    const x1 = x + c.x1;
    const s = 8;
    ctx.strokeStyle = c.color || color;
    ctx.beginPath();
    ctx.moveTo(x0, y + h / 2);
    ctx.lineTo(x0 + s, y);
    ctx.lineTo(x1 - s, y);
    ctx.lineTo(x1, y + h / 2);
    ctx.lineTo(x1 - s, y + h);
    ctx.lineTo(x0 + s, y + h);
    ctx.closePath();
    ctx.stroke();
    if (c.fill) {
      ctx.fillStyle = hexA(c.fill, 0.18);
      ctx.fill();
    }
    text(ctx, c.label, (x0 + x1) / 2, y + h / 2 + size * 0.36, { size, fam: F.mono, weight: 700, color: c.color || color, align: "center" });
  }
  ctx.restore();
}
