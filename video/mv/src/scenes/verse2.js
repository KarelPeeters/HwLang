import { C, W, H, clamp, ease, prog, win, pulse, hexA, lerp, kick } from "../lib.js";
import { F, text, measure, panel, dotGrid, line, glow, rect, drawAnsi, drawCode } from "../draw.js";
import { wordT } from "../lyrics.js";
import { compile } from "../khdl.js";

const PORT_SRC = "module top ports(\n    x: in async int(0..256),\n) {}";

function intervalBar(ctx, x0, x1, y, h, col, alpha, label) {
  rect(ctx, x0, y - h / 2, x1 - x0, h, hexA(col, 0.22), alpha);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = col;
  ctx.lineWidth = 3;
  ctx.shadowColor = col;
  ctx.shadowBlur = 16;
  ctx.strokeRect(x0, y - h / 2, x1 - x0, h);
  ctx.restore();
  if (label) text(ctx, label, (x0 + x1) / 2, y + 12, { size: 32, fam: F.mono, weight: 800, color: col, align: "center", alpha });
}

export const verse2 = {
  a: 111.0,
  b: 134.5,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 8, oy: t * 4 });
    const tBits = wordT("I don't count", "bits");
    const tValues = wordT("I don't count", "values");
    const tComp = wordT("The compiler does");
    const tClog = wordT("The compiler does", "c-log");
    const tSpeed = wordT("The compiler does", "speed");
    const tAdd = wordT("Add two numbers");
    const tCarry = wordT("Add two numbers", "carry");
    const tSqueeze = wordT("Add two numbers", "squeeze");
    const tMis = wordT("Add two numbers", "mismatch");
    const tWrap = wordT("If you want the counter");
    const tMod = wordT("If you want the counter", "modulo");
    const tOne = wordT("A one-flop");
    const tTwo = wordT("Two flops minimum");
    const tTypes = wordT("Two flops minimum", "range");

    // number line
    const NX0 = 200;
    const NX1 = 1720;
    const NY = 640;
    const sc = (v, max) => lerp(NX0, NX1, v / max);

    // ---- A: values, not bits ----
    {
      const a = win(t, 111.2, tAdd - 0.1, 0.4, 0.3);
      if (a > 0) {
        // the old way, struck through
        const po = prog(t, 111.3, 0.4);
        text(ctx, "wire [7:0] x;", 560, 260, { size: 64, fam: F.mono, weight: 700, color: C.muted, align: "center", alpha: a * po });
        const st = prog(t, tBits - 0.05, 0.25);
        if (st > 0) line(ctx, 330, 238, lerp(330, 790, st), 238, C.red, 6, a);
        const pv = prog(t, tValues - 0.1, 0.35, ease.outBack);
        if (pv > 0) {
          ctx.save();
          ctx.translate(1340, 260);
          ctx.scale(pv, pv);
          drawCode(ctx, "x: int(0..256)", -measure(ctx, "x: int(0..256)", 64, F.mono, 400) / 2, 0, { size: 64, alpha: a });
          ctx.restore();
        }
        // number line with an interval
        const pl = prog(t, tValues - 0.2, 0.8, ease.inOutCubic);
        line(ctx, NX0, NY, lerp(NX0, NX1, pl), NY, C.muted, 2, a);
        for (let v = 0; v <= 256; v += 32) {
          const x = sc(v, 256);
          if (x > lerp(NX0, NX1, pl)) break;
          line(ctx, x, NY - 10, x, NY + 10, C.muted, 2, a);
          text(ctx, String(v), x, NY + 44, { size: 22, fam: F.mono, color: C.muted, align: "center", alpha: a });
        }
        const pi = prog(t, tValues + 0.1, 0.9, ease.outCubic);
        if (pi > 0) intervalBar(ctx, sc(0, 256), sc(256 * pi, 256), NY - 60, 50, C.cyan, a, pi > 0.9 ? "0 .. 256" : "");
        // compiler does the clog2 — real lowered Verilog from the real compiler
        const pc = prog(t, tClog - 0.1, 0.3);
        if (pc > 0) {
          const r = compile(PORT_SRC);
          const portLine = (r.verilog.split("\n").find((l) => l.includes("input")) || "").trim();
          panel(ctx, 460, 790, 1000, 120, { alpha: a * pc, title: "khdl → verilog" });
          const n = Math.floor(clamp((t - tClog) / 0.8) * portLine.length);
          drawCode(ctx, portLine.slice(0, n), 500, 870, { size: 34, lang: "verilog", alpha: a * pc });
          const ps = prog(t, tSpeed - 0.1, 0.2, ease.outBack);
          if (ps > 0) {
            ctx.save();
            ctx.translate(1530, 850);
            ctx.scale(ps, ps);
            text(ctx, "$clog2 ✓", 0, 0, { size: 40, fam: F.disp, color: C.green, align: "left", alpha: a });
            text(ctx, "done at elaboration", 0, 36, { size: 22, fam: F.mono, color: C.muted, alpha: a });
            ctx.restore();
          }
        }
      }
    }

    // ---- B: keep the carry, type mismatch, modulo ----
    {
      const a = win(t, tAdd - 0.15, tOne - 0.1, 0.3, 0.3);
      if (a > 0) {
        const NY = 520;
        const max = 512;
        const fuse = prog(t, tAdd + 0.5, 0.7, ease.inOutCubic);
        const wrap = prog(t, tWrap - 0.1, 0.4, ease.inOutCubic);
        const la = a * (1 - wrap);
        line(ctx, NX0, NY, NX1, NY, C.muted, 2, la);
        for (let v = 0; v <= max; v += 64) {
          const x = sc(v, max);
          line(ctx, x, NY - 10, x, NY + 10, C.muted, 2, la);
          text(ctx, String(v), x, NY + 44, { size: 22, fam: F.mono, color: C.muted, align: "center", alpha: la });
        }
        if (wrap < 1) {
          const al = a * (1 - wrap);
          // a and b
          intervalBar(ctx, sc(0, max), sc(255, max), lerp(NY - 260, NY - 70, fuse), 50, C.cyan, al * (1 - fuse * 0.999), "a : uint(8)");
          intervalBar(ctx, lerp(sc(0, max), sc(256, max), fuse), lerp(sc(255, max), sc(511, max), fuse), lerp(NY - 170, NY - 70, fuse), 50, C.magenta, al * (1 - fuse * 0.999), "b : uint(8)");
          if (fuse >= 1) {
            const sq = prog(t, tSqueeze - 0.05, 0.4, ease.inOutCubic);
            intervalBar(ctx, sc(0, max), sc(511, max), NY - 70, 50, t > tMis ? C.red : C.green, al, "a + b : int(0..511)");
            // the carry glows
            const pk = prog(t, tCarry - 0.1, 0.3);
            if (pk > 0 && t < tSqueeze) text(ctx, "carry kept ✓", sc(384, max), NY - 130, { size: 32, fam: F.mono, weight: 800, color: C.green, align: "center", alpha: al * pk });
            // narrow slot pushing down from above
            if (sq > 0) {
              const sy = lerp(NY - 380, NY - 170, sq);
              ctx.save();
              ctx.globalAlpha = al;
              ctx.strokeStyle = C.amber;
              ctx.setLineDash([12, 8]);
              ctx.lineWidth = 4;
              ctx.strokeRect(sc(0, max), sy - 40, sc(256, max) - sc(0, max), 80);
              ctx.setLineDash([]);
              ctx.restore();
              text(ctx, "sum : out uint(8)", sc(128, max), sy + 12, { size: 30, fam: F.mono, weight: 700, color: C.amber, align: "center", alpha: al });
            }
            // the real compiler error
            const pm = prog(t, tMis - 0.05, 0.25, ease.outBack);
            if (pm > 0) {
              ctx.save();
              ctx.translate(W / 2, 680);
              ctx.scale(lerp(0.9, 1, pm), lerp(0.9, 1, pm));
              panel(ctx, -760, -60, 1520, 300, { title: "khdl build", alpha: al * pm });
              drawAnsi(ctx, window.COMPILER.diags.trunc, -720, 10, { size: 20, alpha: al * pm });
              ctx.restore();
            }
          }
        }
        // modulo wrap: a counter on a ring
        if (wrap > 0) {
          const cx = W / 2;
          const cy = 470;
          const R = 230;
          ctx.save();
          ctx.globalAlpha = a * wrap;
          ctx.strokeStyle = C.muted;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(cx, cy, R, 0, Math.PI * 2);
          ctx.stroke();
          for (let k = 0; k < 16; k++) {
            const ang = (k / 16) * Math.PI * 2 - Math.PI / 2;
            line(ctx, cx + Math.cos(ang) * (R - 12), cy + Math.sin(ang) * (R - 12), cx + Math.cos(ang) * (R + 12), cy + Math.sin(ang) * (R + 12), C.muted, 2);
          }
          text(ctx, "0", cx, cy - R - 30, { size: 26, fam: F.mono, color: C.text, align: "center" });
          text(ctx, "255", cx - 64, cy - R - 30, { size: 26, fam: F.mono, color: C.muted, align: "center" });
          const pos = (t - tWrap) * 1.6;
          const ang = pos * Math.PI * 2 - Math.PI / 2;
          glow(ctx, C.cyan, 30, () => {
            ctx.fillStyle = C.cyan;
            ctx.beginPath();
            ctx.arc(cx + Math.cos(ang) * R, cy + Math.sin(ang) * R, 16, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = C.cyan;
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.arc(cx, cy, R, -Math.PI / 2, ang);
            ctx.stroke();
          });
          text(ctx, String(Math.floor(((pos % 1) + 1) % 1 * 256)), cx, cy + 24, { size: 80, fam: F.mono, weight: 800, color: C.text, align: "center" });
          ctx.restore();
          const pm = prog(t, tMod - 0.1, 0.3, ease.outBack);
          if (pm > 0) {
            ctx.save();
            ctx.translate(W / 2, 830);
            ctx.scale(pm, pm);
            const s = "count = (count + 1) % 256;";
            drawCode(ctx, s, -measure(ctx, s, 52, F.mono, 400) / 2, 0, { size: 52, alpha: a * wrap });
            text(ctx, "wrapping is something you say, not something that happens", 0, 60, { size: 26, fam: F.sans, color: C.muted, align: "center", alpha: a * wrap });
            ctx.restore();
          }
        }
      }
    }

    // ---- C: synchronizer must have two flops ----
    {
      const a = win(t, tOne - 0.2, 134.6, 0.3, 0.3);
      if (a > 0) {
        const two = t >= tTwo - 0.05;
        const n = two ? 2 : 1;
        const s = 150;
        const gap = 110;
        const total = n * s + (n - 1) * gap;
        const x0 = W / 2 - total / 2;
        const y0 = 230;
        line(ctx, x0 - 260, y0 + s / 2, x0 + total + 260, y0 + s / 2, C.dim, 4, a);
        text(ctx, "async", x0 - 260, y0 + s / 2 - 20, { size: 26, fam: F.mono, weight: 700, color: C.amber, alpha: a });
        text(ctx, "sync(clk)", x0 + total + 260, y0 + s / 2 - 20, { size: 26, fam: F.mono, weight: 700, color: C.cyan, align: "right", alpha: a });
        for (let i = 0; i < n; i++) {
          const p = i === 1 ? prog(t, tTwo - 0.05, 0.3, ease.outBack) : 1;
          ctx.save();
          ctx.translate(x0 + i * (s + gap) + s / 2, y0 + s / 2);
          ctx.scale(p, p);
          const col = two ? C.green : C.red;
          glow(ctx, col, 30, () => {
            rect(ctx, -s / 2, -s / 2, s, s, "#121620", a);
            ctx.strokeStyle = col;
            ctx.lineWidth = 4;
            ctx.globalAlpha = a;
            ctx.strokeRect(-s / 2, -s / 2, s, s);
          });
          text(ctx, "FF", 0, 18, { size: 54, fam: F.disp, color: C.text, align: "center", alpha: a });
          ctx.restore();
        }
        if (!two) {
          // the real error for N=1
          const pd = prog(t, tOne + 0.4, 0.3);
          panel(ctx, 180, 470, 1560, 300, { title: "khdl build", alpha: a * pd });
          drawAnsi(ctx, window.COMPILER.diags.sync1, 220, 550, { size: 22, alpha: a * pd });
        } else {
          const pc = prog(t, tTwo, 0.3);
          const s1 = "module cdc_sync_ff(N: int(2..) = 2) ports(...)";
          ctx.save();
          ctx.translate(W / 2 - measure(ctx, s1, 46, F.mono, 400) / 2, 600);
          drawCode(ctx, s1, 0, 0, { size: 46, alpha: a * pc });
          ctx.restore();
          const pr = prog(t, tTypes - 0.1, 0.3, ease.outBack);
          const cw = measure(ctx, "M", 46, F.mono, 400);
          const hx = W / 2 - measure(ctx, s1, 46, F.mono, 400) / 2 + 19 * cw;
          if (pr > 0) {
            ctx.save();
            ctx.globalAlpha = a * pr;
            ctx.strokeStyle = C.green;
            ctx.lineWidth = 4;
            ctx.shadowColor = C.green;
            ctx.shadowBlur = 20;
            ctx.strokeRect(hx - 10, 552, 8 * cw + 20, 66);
            ctx.restore();
            text(ctx, "a one-flop synchronizer is a type error", W / 2, 720, { size: 32, fam: F.sans, weight: 700, color: C.green, align: "center", alpha: a * pr });
          }
        }
      }
    }
  },
};

// ---------- pre-chorus 2: comb / clocked / reg wire / undef ----------
const PC2_SRC = `comb {
    sum = a + b;
}

clocked(clk, async rst) {
    reg wire count = 0;
    count = (count + 1) % 256;

    reg scratch: uint(8) = undef;
}`;

export const pre2 = {
  a: 134.3,
  b: 146.2,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 8 });
    const tComb = wordT("Comb or clocked");
    const tClocked = wordT("Comb or clocked", "clocked");
    const tTwo = wordT("Comb or clocked", "two");
    const tFlop = wordT("Want a flop", "flop");
    const tReg = wordT("Want a flop", "reg");
    const tSneak = wordT("Want a flop", "sneak");
    const tSkip = wordT("Skip the reset");
    const tUndef = wordT("Skip the reset", "equals");
    const tLoud = wordT("Skip the reset", "loud");
    const tForgot = wordT("I didn't forget");
    const tDecl = wordT("I didn't forget", "declared");
    const tProud = wordT("I didn't forget", "proud");
    const a = win(t, 134.3, 146.2, 0.3, 0.35);
    // two kinds of blocks, as two big shapes, then they become the code
    const toCode = prog(t, tFlop - 0.3, 0.6, ease.inOutCubic);
    const pc = prog(t, tComb - 0.05, 0.25, ease.outBack);
    const pk = prog(t, tClocked - 0.05, 0.25, ease.outBack);
    if (toCode < 1) {
      const al = a * (1 - toCode);
      ctx.save();
      ctx.translate(620, 460);
      ctx.scale(pc, pc);
      panel(ctx, -300, -200, 600, 400, { alpha: al, fill: "#101826" });
      text(ctx, "comb", 0, 30, { size: 110, fam: F.mono, weight: 800, color: C.cyan, align: "center", alpha: al });
      text(ctx, "every path drives it, or none", 0, 120, { size: 24, fam: F.sans, color: C.muted, align: "center", alpha: al });
      ctx.restore();
      ctx.save();
      ctx.translate(1300, 460);
      ctx.scale(pk, pk);
      panel(ctx, -300, -200, 600, 400, { alpha: al, fill: "#160f22" });
      text(ctx, "clocked", 0, 30, { size: 110, fam: F.mono, weight: 800, color: C.magenta, align: "center", alpha: al });
      text(ctx, "registers live here, nowhere else", 0, 120, { size: 24, fam: F.sans, color: C.muted, align: "center", alpha: al });
      ctx.globalAlpha = al;
      ctx.strokeStyle = C.magenta;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(-300, 140);
      ctx.lineTo(-270, 160);
      ctx.lineTo(-300, 180);
      ctx.stroke();
      ctx.restore();
      const p2 = prog(t, tTwo - 0.05, 0.25, ease.outBack);
      if (p2 > 0) {
        ctx.save();
        ctx.translate(W / 2, 780);
        ctx.scale(p2, p2);
        text(ctx, "only two.", 0, 0, { size: 60, fam: F.disp, color: C.text, align: "center", alpha: al });
        ctx.restore();
      }
    }
    if (toCode > 0) {
      const al = a * toCode;
      const X = 420;
      const Y = 140;
      panel(ctx, X, Y, 1080, 640, { title: "counter.kh", alpha: al });
      const lineAlpha = (ln) => {
        if (ln <= 2) return 0.55;
        if (ln === 4 || ln === 5 || ln === 6) return t > tReg - 0.2 ? 1 : 0.45;
        if (ln === 8) return t > tUndef - 0.2 ? 1 : 0.25;
        return 0.8;
      };
      drawCode(ctx, PC2_SRC, X + 50, Y + 100, { size: 36, alpha: al, lineAlpha });
      const cw = measure(ctx, "M", 36, F.mono, 400);
      const lh = 36 * 1.45;
      // flop stamped onto `reg wire count`
      const pf = prog(t, tReg, 0.25, ease.outBack);
      if (pf > 0) {
        ctx.save();
        ctx.translate(X + 50 + 28 * cw, Y + 100 + 5 * lh - 12);
        ctx.scale(pf, pf);
        rect(ctx, -40, -40, 80, 80, "#121620", al);
        ctx.strokeStyle = C.magenta;
        ctx.lineWidth = 3;
        ctx.globalAlpha = al;
        ctx.strokeRect(-40, -40, 80, 80);
        text(ctx, "FF", 0, 12, { size: 30, fam: F.disp, color: C.text, align: "center", alpha: al });
        ctx.restore();
        const ps = prog(t, tSneak - 0.1, 0.3);
        text(ctx, "no `reg wire`, no flop. a clocked block can't drive a plain port.", X + 540, Y + 700, { size: 24, fam: F.mono, color: C.muted, align: "center", alpha: al * ps });
      }
      // the proud undef
      const pu = prog(t, tUndef - 0.05, 0.25);
      if (pu > 0) {
        const ux = X + 50 + 27 * cw;
        const uy = Y + 100 + 8 * lh;
        ctx.save();
        ctx.globalAlpha = al * pu;
        ctx.strokeStyle = C.amber;
        ctx.lineWidth = 3;
        ctx.strokeRect(ux - 8, uy - 40, 7 * cw + 16, 54);
        ctx.restore();
        // rosette
        const pr = prog(t, tDecl - 0.1, 0.35, ease.outElastic);
        if (pr > 0) {
          ctx.save();
          ctx.translate(1600, 600);
          ctx.rotate(0.12 + Math.sin(t * 3) * 0.04);
          ctx.scale(pr, pr);
          ctx.globalAlpha = al;
          // ribbons
          ctx.fillStyle = "#b8862b";
          ctx.beginPath();
          ctx.moveTo(-50, 40);
          ctx.lineTo(-90, 200);
          ctx.lineTo(-40, 170);
          ctx.lineTo(-10, 210);
          ctx.lineTo(0, 50);
          ctx.fill();
          ctx.beginPath();
          ctx.moveTo(50, 40);
          ctx.lineTo(90, 200);
          ctx.lineTo(40, 170);
          ctx.lineTo(10, 210);
          ctx.lineTo(0, 50);
          ctx.fill();
          ctx.fillStyle = C.amber;
          ctx.beginPath();
          for (let k = 0; k < 24; k++) {
            const r = k % 2 ? 105 : 120;
            const ang = (k / 24) * Math.PI * 2;
            k ? ctx.lineTo(Math.cos(ang) * r, Math.sin(ang) * r) : ctx.moveTo(Math.cos(ang) * r, Math.sin(ang) * r);
          }
          ctx.fill();
          ctx.fillStyle = "#1a1200";
          ctx.beginPath();
          ctx.arc(0, 0, 88, 0, Math.PI * 2);
          ctx.fill();
          text(ctx, "undef", 0, -4, { size: 40, fam: F.mono, weight: 800, color: C.amber, align: "center" });
          text(ctx, "DECLARED", 0, 34, { size: 22, fam: F.disp, color: C.amber, align: "center", spacing: 3 });
          ctx.restore();
        }
        const pp = prog(t, tProud - 0.1, 0.3, ease.outBack);
        if (pp > 0) {
          const f = Math.exp(-Math.max(0, t - tProud) * 4);
          rect(ctx, 0, 0, W, H, C.amber, 0.08 * f);
        }
      }
    }
  },
};
