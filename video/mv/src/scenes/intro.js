import { C, W, H, clamp, ease, prog, win, beatInfo, pulse, A, rnd, hexA, lerp } from "../lib.js";
import { F, text, measure, drawAnsi, panel, dotGrid, line, glow, rect } from "../draw.js";
import { wordT } from "../lyrics.js";

// clock value at absolute time tau: high during first half of each beat
function clkAt(tau) {
  const b = A.data.beats;
  if (tau < b[0]) return 0;
  const { phase } = beatInfo(tau);
  return phase < 0.5 ? 1 : 0;
}

// irregular async signal
function asyncAt(tau) {
  return rnd(Math.floor(tau * 3.1), 77) > 0.5 ? 1 : 0;
}

// scope trace: signal history scrolling left, "now" at headX
function trace(ctx, t, { y, h = 70, headX = W * 0.72, span = 4.5, sig = clkAt, color = C.cyan, alpha = 1, lw = 3, x0 = 0 }) {
  if (alpha <= 0) return;
  const pxPerS = (headX - x0) / span;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.beginPath();
  const steps = 900;
  let prev = null;
  for (let i = 0; i <= steps; i++) {
    const x = x0 + ((headX - x0) * i) / steps;
    const tau = t - (headX - x) / pxPerS;
    if (tau < 0) continue;
    const v = sig(tau);
    const yy = y - v * h;
    if (prev === null) ctx.moveTo(x, yy);
    else if (yy !== prev) ctx.lineTo(x, prev), ctx.lineTo(x, yy);
    else ctx.lineTo(x, yy);
    prev = yy;
  }
  ctx.stroke();
  // pen head
  if (prev !== null) {
    ctx.fillStyle = "#fff";
    ctx.shadowBlur = 30;
    ctx.beginPath();
    ctx.arc(headX, prev, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function flop(ctx, x, y, s, { letter = "", alpha = 1, lit = 0, color = C.cyan, letterY = 0 }) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "#0f131c";
  ctx.strokeStyle = lit > 0 ? hexA(color, 0.4 + lit * 0.6) : C.rule;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = lit * 40;
  ctx.fillRect(x, y, s, s);
  ctx.strokeRect(x, y, s, s);
  ctx.shadowBlur = 0;
  // clock triangle
  ctx.beginPath();
  ctx.moveTo(x, y + s - 46);
  ctx.lineTo(x + 18, y + s - 34);
  ctx.lineTo(x, y + s - 22);
  ctx.stroke();
  text(ctx, "D", x + 12, y + 34, { size: 18, fam: F.mono, color: C.muted });
  text(ctx, "Q", x + s - 26, y + 34, { size: 18, fam: F.mono, color: C.muted });
  if (letter) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, s, s);
    ctx.clip();
    glow(ctx, color, 30 * lit, () => {
      text(ctx, letter, x + s / 2, y + s / 2 + 58 + letterY, { size: 160, fam: F.disp, weight: 400, color: C.text, align: "center" });
    });
    ctx.restore();
  }
  ctx.restore();
}

export const intro = {
  a: 0,
  b: 27.2,
  lyrics: "none",
  render(ctx, t) {
    const tErr = wordT("Error", "error");
    const tInv = wordT("Error", "invalid");
    const tAsync = wordT("Async", "async");
    const tSync = wordT("Async", "sync");
    const out = 1 - prog(t, 25.6, 1.4, ease.inCubic);

    dotGrid(ctx, t, { alpha: 0.6 * prog(t, 0.2, 2) * out, ox: -t * 12 });

    // --- 0 .. 11: scope trace + error headline ---
    const p1 = 1 - prog(t, 6.0, 0.6);
    trace(ctx, t, { y: 610, h: 60, alpha: prog(t, 0.0, 0.8) * p1, headX: lerp(W * 0.5, W * 0.72, prog(t, 0, 3)) });

    // headline
    if (t > tErr - 0.1 && t < 12) {
      const a = win(t, tErr - 0.05, 11.6, 0.05, 0.6);
      const up = prog(t, 6.2, 0.9, ease.inOutCubic);
      const y = lerp(470, 200, up);
      const size = lerp(84, 52, up);
      const s1 = "error: ";
      const s2 = "invalid domain crossing";
      const n2 = Math.floor(clamp((t - tInv) / 1.6) * s2.length);
      const w1 = measure(ctx, s1, size, F.mono, 800);
      const w2 = measure(ctx, s2, size, F.mono, 800);
      const x0 = W / 2 - (w1 + w2) / 2;
      const jit = pulse(t, 14) * 3;
      glow(ctx, C.red, 30, () => text(ctx, s1, x0 + jit, y, { size, fam: F.mono, weight: 800, color: C.red, alpha: a }));
      text(ctx, s2.slice(0, n2), x0 + w1, y, { size, fam: F.mono, weight: 800, color: C.text, alpha: a });
      if (n2 < s2.length || Math.floor(t * 2.5) % 2) rect(ctx, x0 + w1 + measure(ctx, s2.slice(0, n2), size, F.mono, 800) + 4, y - size * 0.8, size * 0.5, size * 0.95, C.text, a * 0.8);
    }

    // real diagnostic from the compiler
    {
      const a = win(t, 6.4, 11.4, 0.5, 0.5);
      if (a > 0) {
        const d = window.COMPILER.diags.cdc;
        const y = 290 + (1 - ease.outCubic(clamp((t - 6.4) / 0.8))) * 60;
        panel(ctx, 300, y, 1320, 470, { title: "khdl build", alpha: a });
        drawAnsi(ctx, d, 340, y + 84, { size: 21, alpha: a, chars: Math.floor((t - 6.5) * 900) });
      }
    }

    // async vs sync lanes, highlighted on the sung words
    {
      const a = win(t, 8.4, 14.5, 0.6, 1.2);
      if (a > 0) {
        const ya = 880;
        const ys = 990;
        const hiA = prog(t, tAsync, 0.2) * (1 - prog(t, tSync, 0.3) * 0.5);
        const hiS = prog(t, tSync, 0.2);
        text(ctx, "async", 120, ya - 18, { size: 26, fam: F.mono, weight: 700, color: C.amber, alpha: a * (0.4 + 0.6 * hiA) });
        text(ctx, "sync(clk)", 120, ys - 18, { size: 26, fam: F.mono, weight: 700, color: C.cyan, alpha: a * (0.4 + 0.6 * hiS) });
        trace(ctx, t, { y: ya, h: 50, sig: asyncAt, color: C.amber, alpha: a * (0.35 + 0.65 * hiA), headX: W - 140, x0: 330, span: 5 });
        trace(ctx, t, { y: ys, h: 50, sig: clkAt, color: C.cyan, alpha: a * (0.35 + 0.65 * hiS), headX: W - 140, x0: 330, span: 5 });
        // crossing arrow + metastable glitch
        const pa = prog(t, tSync, 0.35);
        if (pa > 0) {
          const x = W - 300;
          line(ctx, x, ya + 8, x, lerp(ya + 8, ys - 62, pa), C.red, 3, a);
          if (pa >= 1) {
            ctx.save();
            ctx.globalAlpha = a;
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 3;
            ctx.shadowColor = C.red;
            ctx.shadowBlur = 20;
            ctx.beginPath();
            for (let i = 0; i < 40; i++) {
              const yy = ys - 36 + Math.sin(i * 1.7 + t * 40) * 22 * Math.exp(-i / 18);
              i ? ctx.lineTo(x + i * 4, yy) : ctx.moveTo(x, yy);
            }
            ctx.stroke();
            ctx.restore();
            text(ctx, "metastable", x + 20, ys - 74, { size: 20, fam: F.mono, color: C.red, alpha: a });
          }
        }
      }
    }

    // --- 11 .. 27: shift register spells K H D L ---
    {
      const a = win(t, 11.3, 27.2, 0.6, 0.01) * out;
      if (a > 0) {
        ctx.save();
        const push = 1 + 0.06 * ease.inOutQuad(clamp((t - 16) / 10)) + 0.012 * pulse(t, 7);
        ctx.translate(W / 2, H / 2);
        ctx.scale(push, push);
        ctx.translate(-W / 2, -H / 2);
        const b = A.data.beats;
        // shift on every second beat from ~12s
        const first = b.findIndex((x) => x >= 12.0);
        const shiftTimes = [0, 1, 2, 3].map((k) => b[first + k * 2]);
        const feed = ["L", "D", "H", "K"];
        const s = 230;
        const gap = 70;
        const total = 4 * s + 3 * gap;
        const x0 = W / 2 - total / 2;
        const y0 = 300;
        const nShift = shiftTimes.filter((x) => t >= x).length;
        // wires between flops
        for (let i = 0; i < 5; i++) {
          const xa = i === 0 ? x0 - 200 : x0 + i * (s + gap) - gap;
          const xb = i === 4 ? x0 + total + 200 : x0 + i * (s + gap);
          line(ctx, xa, y0 + s / 2, xb, y0 + s / 2, C.dim, 3, a);
        }
        for (let i = 0; i < 4; i++) {
          // position i holds feed[nShift-1-i] (shift right)
          const k = nShift - 1 - i;
          const letter = k >= 0 ? feed[k] : "";
          const since = k >= 0 ? t - shiftTimes[nShift - 1] : 9;
          const lit = Math.exp(-since * 3.5);
          const letterY = (1 - ease.outBack(clamp(since / 0.25), 2)) * -s * 0.6;
          const bi = beatInfo(t).i;
          const chase = t > shiftTimes[3] + 0.5 && bi % 4 === i ? pulse(t, 6) * 0.8 : 0;
          flop(ctx, x0 + i * (s + gap), y0, s, { letter, alpha: a, lit: lit + 0.15 * pulse(t, 5) + chase, letterY });
        }
        // incoming letter waiting at the input
        if (nShift < 4) {
          const nl = feed[nShift];
          text(ctx, nl, x0 - 120, y0 + s / 2 + 40, { size: 110, fam: F.disp, color: C.muted, align: "center", alpha: a * 0.5 });
        }
        // clock lane under the register
        trace(ctx, t, { y: y0 + s + 120, h: 40, alpha: a * 0.85, headX: x0 + total, x0, span: 3.2 });
        text(ctx, "clk", x0 - 70, y0 + s + 112, { size: 22, fam: F.mono, color: C.cyan, alpha: a * 0.8 });

        // title words slam on beats after register is full
        const titleBeat = b.findIndex((x) => x >= shiftTimes[3] + 1.6);
        const words = ["INVALID", "DOMAIN", "CROSSING"];
        const ty = 860;
        const sizes = 104;
        const widths = words.map((w) => measure(ctx, w, sizes, F.disp, 400, 4));
        const totalW = widths.reduce((s, w) => s + w, 0) + 2 * 40;
        let x = W / 2 - totalW / 2;
        words.forEach((w, i) => {
          const tb = b[titleBeat + i * 2];
          const p = clamp((t - tb) / 0.22);
          if (p > 0) {
            const sc = lerp(1.6, 1, ease.outExpo(p));
            ctx.save();
            ctx.translate(x + widths[i] / 2, ty);
            ctx.scale(sc, sc);
            text(ctx, w, 0, 0, { size: sizes, fam: F.disp, color: i === 2 ? C.red : C.text, align: "center", alpha: a * clamp(p * 2), spacing: 4 });
            ctx.restore();
          }
          x += widths[i] + 40;
        });
        const subT = b[titleBeat + 6];
        text(ctx, "a song about KHDL  ·  a hardware description language that checks what you meant", W / 2, 940, { size: 26, fam: F.sans, weight: 500, color: C.muted, align: "center", alpha: a * prog(t, subT, 0.6), spacing: 1 });
        ctx.restore();
      }
    }
  },
};
