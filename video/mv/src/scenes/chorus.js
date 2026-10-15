import { C, W, H, clamp, ease, prog, win, pulse, hexA, lerp, rnd, kick, beatInfo, A } from "../lib.js";
import { F, text, measure, panel, dotGrid, line, glow, rect, drawAnsi, wave } from "../draw.js";
import { lines } from "../lyrics.js";

const norm = (w) => w.toLowerCase().replace(/[^a-z0-9]/g, "");

function chorusGroups() {
  const groups = [];
  let cur = null;
  for (const l of lines()) {
    const isCh = /^(Final )?Chorus/.test(l.section || "");
    if (isCh) {
      if (!cur) groups.push((cur = []));
      cur.push(l);
    } else cur = null;
  }
  return groups;
}

function tokT(line, word) {
  const tk = line.toks.find((tk) => norm(tk.text).startsWith(norm(word)));
  return tk ? tk.s : line.s;
}

function slash(ctx, x, y, s, p, color = C.red) {
  if (p <= 0) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 14;
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 30;
  ctx.beginPath();
  ctx.moveTo(x - s, y - s);
  ctx.lineTo(x - s + 2 * s * p, y - s + 2 * s * p);
  ctx.stroke();
  ctx.restore();
}

function lanes(ctx, t, accent, alpha, boost = 1) {
  const cols = [accent, C.cyan, C.magenta, C.amber, accent, C.violet];
  const k = kick(t);
  for (let i = 0; i < 6; i++) {
    const y = 330 + i * 120;
    const per = 120 + i * 30;
    wave(ctx, -50, y, W + 100, 34, { period: per, phase: (t * (80 + i * 25)) % per, color: cols[i], lw: 2, alpha: alpha * (0.07 + 0.1 * k) * boost, glowAmt: 0 });
  }
}

function lineVisual(ctx, t, line, accent, idx, nextS, intense) {
  const txt = line.text.toLowerCase();
  const a = win(t, line.s - 0.25, Math.min(line.e + 0.5, nextS - 0.02), 0.2, 0.15);
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  if (txt.includes("latches")) {
    const pl = prog(t, line.s - 0.1, 0.3, ease.outBack);
    // latch card
    ctx.save();
    ctx.translate(640, 640);
    ctx.scale(pl, pl);
    panel(ctx, -230, -140, 460, 280, { fill: "#1a1016" });
    text(ctx, "LATCH", 0, 30, { size: 90, fam: F.disp, color: C.red, align: "center" });
    ctx.restore();
    const ps = prog(t, tokT(line, "sensitivity") - 0.1, 0.3, ease.outBack);
    ctx.save();
    ctx.translate(1280, 640);
    ctx.scale(ps, ps);
    panel(ctx, -260, -140, 520, 280, {});
    text(ctx, "@(a or b)", 0, 24, { size: 70, fam: F.mono, weight: 700, color: C.amber, align: "center" });
    ctx.restore();
    slash(ctx, 640, 640, 170, prog(t, tokT(line, "latches"), 0.2));
    slash(ctx, 1280, 640, 170, prog(t, tokT(line, "lists"), 0.2));
  } else if (txt.includes("equals sign")) {
    const k = kick(t);
    const p = prog(t, line.s - 0.1, 0.3, ease.outBack);
    ctx.save();
    ctx.translate(W / 2, 640);
    ctx.scale(p * (1 + 0.05 * k), p * (1 + 0.05 * k));
    glow(ctx, accent, 60, () => text(ctx, "=", 0, 150, { size: 520, fam: F.mono, weight: 800, color: C.text, align: "center" }));
    ctx.restore();
    const pc = prog(t, tokT(line, "does"), 0.3);
    text(ctx, "count = count + 1;   // the next line sees the new value", W / 2, 900, { size: 30, fam: F.mono, color: C.muted, align: "center", alpha: a * pc });
  } else if (txt.includes("knows its clock")) {
    const doms = [
      ["sync(clk_a)", C.cyan],
      ["sync(clk_b)", C.magenta],
      ["async", C.amber],
    ];
    const tc = tokT(line, "clock");
    const tf = tokT(line, "flop");
    const tr = tokT(line, "reset");
    doms.forEach(([name, col], i) => {
      const y = 520 + i * 140;
      const pw = prog(t, line.s - 0.1 + i * 0.08, 0.4, ease.outCubic);
      line_(ctx, 520, y, lerp(520, 1500, pw), y, col, 5, a);
      const pt = prog(t, tc - 0.1 + i * 0.1, 0.25, ease.outBack);
      if (pt > 0) {
        ctx.save();
        ctx.translate(420, y);
        ctx.scale(pt, pt);
        const w = measure(ctx, name, 30, F.mono, 700) + 30;
        ctx.fillStyle = hexA(col, 0.16);
        ctx.fillRect(-w / 2, -26, w, 52);
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.strokeRect(-w / 2, -26, w, 52);
        text(ctx, name, 0, 11, { size: 30, fam: F.mono, weight: 700, color: col, align: "center" });
        ctx.restore();
      }
      const pf = prog(t, tf - 0.05 + i * 0.08, 0.25, ease.outBack);
      if (pf > 0 && i < 2) {
        ctx.save();
        ctx.translate(1100, y);
        ctx.scale(pf, pf);
        rect(ctx, -60, -60, 120, 120, "#121620");
        ctx.strokeStyle = col;
        ctx.lineWidth = 3;
        ctx.strokeRect(-60, -60, 120, 120);
        text(ctx, "D  Q", 0, -26, { size: 20, fam: F.mono, color: C.muted, align: "center" });
        const pr = prog(t, tr - 0.05 + i * 0.08, 0.25, ease.outBack);
        if (pr > 0) {
          ctx.scale(pr, pr);
          rect(ctx, -44, 18, 88, 32, col);
          text(ctx, "rst", 0, 42, { size: 22, fam: F.mono, weight: 800, color: "#000", align: "center" });
        }
        ctx.restore();
      }
    });
  } else if (txt.includes("crossing")) {
    const tCatch = tokT(line, "catches");
    const pw = prog(t, line.s, 0.8, ease.inOutCubic);
    // cyan wire trying to cross into magenta domain
    ctx.save();
    ctx.globalAlpha = a;
    rect(ctx, 0, 420, W / 2, 440, C.cyan, 0.05 * a);
    rect(ctx, W / 2, 420, W / 2, 440, C.magenta, 0.05 * a);
    text(ctx, "sync(clk_a)", 120, 470, { size: 30, fam: F.mono, weight: 700, color: C.cyan });
    text(ctx, "sync(clk_b)", W - 120, 470, { size: 30, fam: F.mono, weight: 700, color: C.magenta, align: "right" });
    const xEnd = lerp(200, 1500, pw);
    const stopX = W / 2 - 10;
    const caught = t > tCatch;
    line_(ctx, 200, 640, Math.min(xEnd, caught ? stopX : xEnd), 640, C.cyan, 8, a);
    if (caught) {
      const p = prog(t, tCatch, 0.25, ease.outBack);
      ctx.save();
      ctx.translate(stopX + 10, 640);
      ctx.scale(p, p);
      glow(ctx, C.red, 50, () => {
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 18;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(-60, -60);
        ctx.lineTo(60, 60);
        ctx.moveTo(60, -60);
        ctx.lineTo(-60, 60);
        ctx.stroke();
      });
      ctx.restore();
      const pd = prog(t, tCatch + 0.15, 0.3);
      const diag = window.COMPILER.diags.cdc.split("\n").slice(0, 1).join("\n");
      panel(ctx, 440, 760, 1040, 80, { alpha: a * pd });
      drawAnsi(ctx, diag, 480, 812, { size: 30, alpha: a * pd });
    }
    ctx.restore();
  } else if (/k! h! d! l!/i.test(line.text)) {
    // the four letters, full screen on their onsets
    const letters = line.toks.filter((tk) => /^[KHDL]!$/.test(tk.text));
    const after = letters.length ? letters[letters.length - 1].e : line.s;
    for (let i = 0; i < letters.length; i++) {
      const tk = letters[i];
      const next = letters[i + 1] ? letters[i + 1].s : after;
      const p = clamp((t - tk.s + 0.03) / 0.12);
      if (p <= 0 || t > next) continue;
      const sc = lerp(1.35, 1, ease.outExpo(p));
      const flash = Math.exp(-(t - tk.s) * 9);
      rect(ctx, 0, 0, W, H, i % 2 ? accent : C.text, 0.12 * flash);
      ctx.save();
      ctx.translate(W / 2, H / 2 + 40);
      ctx.scale(sc, sc);
      const lc = intense ? [C.cyan, C.magenta, C.amber, C.green][i % 4] : C.text;
      glow(ctx, intense ? lc : accent, 80 * flash, () => text(ctx, tk.text[0], 0, 230, { size: 680, fam: F.disp, color: lc, align: "center" }));
      ctx.restore();
    }
    // payoff visual after the letters
    if (t > after) {
      const pa = prog(t, after, 0.3);
      if (txt.includes("range")) {
        const tb = tokT(line, "bits");
        text(ctx, "int(0..256)", 560, 640, { size: 88, fam: F.mono, weight: 800, color: C.text, align: "center", alpha: pa });
        text(ctx, "→", 960, 640, { size: 88, fam: F.mono, weight: 800, color: accent, align: "center", alpha: prog(t, after + 0.4, 0.3) });
        for (let b = 0; b < 8; b++) {
          const pb = prog(t, tb - 0.35 + b * 0.05, 0.2, ease.outBack);
          if (pb <= 0) continue;
          ctx.save();
          ctx.translate(1110 + b * 78, 610);
          ctx.scale(pb, pb);
          rect(ctx, -32, -40, 64, 80, hexA(accent, 0.2));
          ctx.strokeStyle = accent;
          ctx.lineWidth = 3;
          ctx.strokeRect(-32, -40, 64, 80);
          text(ctx, String(7 - b), 0, 16, { size: 38, fam: F.mono, weight: 800, color: accent, align: "center" });
          ctx.restore();
        }
        text(ctx, "picked by the compiler", 1384, 720, { size: 28, fam: F.mono, color: C.muted, align: "center", alpha: prog(t, tb, 0.3) });
      } else if (txt.includes("fits")) {
        const tf = tokT(line, "fits");
        const tcmp = tokT(line, "compiles");
        // slot
        const sx = W / 2;
        const sy = 780;
        ctx.save();
        ctx.globalAlpha = pa;
        ctx.strokeStyle = C.muted;
        ctx.lineWidth = 4;
        ctx.setLineDash([14, 10]);
        ctx.strokeRect(sx - 260, sy - 70, 520, 140);
        ctx.setLineDash([]);
        text(ctx, "out uint(0..16)", sx, sy + 120, { size: 30, fam: F.mono, color: C.muted, align: "center" });
        ctx.restore();
        const fall = clamp((t - tcmp) / Math.max(0.2, tf - tcmp));
        const by = lerp(330, sy, ease.inQuad(fall));
        const ok = t > tf;
        ctx.save();
        ctx.translate(sx, by);
        rect(ctx, -250, -62, 500, 124, ok ? hexA(C.green, 0.3) : "#1b2130", pa);
        ctx.strokeStyle = ok ? C.green : accent;
        ctx.lineWidth = 4;
        ctx.globalAlpha = pa;
        ctx.strokeRect(-250, -62, 500, 124);
        text(ctx, "a + b : int(0..15)", 0, 14, { size: 40, fam: F.mono, weight: 800, color: ok ? C.green : C.text, align: "center", alpha: pa });
        ctx.restore();
        if (ok) {
          const f = Math.exp(-(t - tf) * 5);
          rect(ctx, 0, 0, W, H, C.green, 0.15 * f);
          text(ctx, "✓ fits", sx + 380, sy + 18, { size: 60, fam: F.disp, color: C.green, alpha: prog(t, tf, 0.2) });
        }
      } else {
        // "if it compiles..." — the compile spinner holds its breath
        const sp = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[Math.floor(t * 12) % 10];
        text(ctx, `${sp} compiling`, W / 2, 660, { size: 64, fam: F.mono, weight: 700, color: C.muted, align: "center", alpha: pa });
      }
    }
  }
  ctx.restore();
}

function line_(ctx, x1, y1, x2, y2, col, lw, alpha) {
  ctx.save();
  ctx.shadowColor = col;
  ctx.shadowBlur = 16;
  line(ctx, x1, y1, x2, y2, col, lw, alpha);
  ctx.restore();
}

function makeChorus(gi, a, b, accent, tail, intense = false) {
  return {
    a,
    b,
    lyrics: (t) => ({ style: "center", y: 210, size: 70 }),
    render(ctx, t) {
      const g = chorusGroups()[gi];
      dotGrid(ctx, t, { alpha: 0.45, ox: -t * 30 });
      lanes(ctx, t, accent, win(t, a, b, 0.5, 0.5), intense ? 2.2 : 1);
      if (intense) {
        // the camera rides the kick
        const k = kick(t);
        ctx.translate(W / 2, H / 2);
        ctx.scale(1 + 0.025 * k, 1 + 0.025 * k);
        ctx.rotate(Math.sin(t * 1.3) * 0.004);
        ctx.translate(-W / 2, -H / 2);
      }
      // beat flash on the downbeat entry
      const f0 = Math.exp(-Math.max(0, t - g[0].s) * 4);
      if (t >= g[0].s - 0.05) rect(ctx, 0, 0, W, H, accent, 0.18 * f0);
      for (let i = 0; i < g.length; i++) lineVisual(ctx, t, g[i], accent, i, g[i + 1] ? g[i + 1].s : 1e9, intense);
      if (tail) tail(ctx, t, g, accent);
    },
  };
}

function wordmarkTail(ctx, t, g, accent) {
  const t0 = g[g.length - 1].e + 0.5;
  const a = win(t, t0, 111.6, 0.4, 0.5);
  if (a <= 0) return;
  const k = kick(t);
  ctx.save();
  ctx.translate(W / 2, H / 2);
  const sc = 1 + 0.03 * k + (t - t0) * 0.02;
  ctx.scale(sc, sc);
  glow(ctx, accent, 40 + 40 * k, () => text(ctx, "KHDL", 0, 110, { size: 340, fam: F.disp, color: C.text, align: "center", alpha: a, spacing: 20 }));
  text(ctx, "you give the range  ·  it picks the bits", 0, 220, { size: 38, fam: F.sans, weight: 500, color: C.muted, align: "center", alpha: a, spacing: 2 });
  ctx.restore();
}

export const chorus1 = makeChorus(0, 82.0, 111.2, C.cyan, wordmarkTail);
export const chorus2 = makeChorus(1, 145.7, 168.7, C.magenta);
export const chorus3 = makeChorus(2, 283.5, 310.6, C.green, null, true);
