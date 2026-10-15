import { C, W, H, clamp, ease, prog, win, pulse, hexA, lerp, rnd, kick, beatInfo, A } from "../lib.js";
import { F, text, measure, panel, dotGrid, line, glow, rect, drawAnsi, drawCode, ansiPlain, wave } from "../draw.js";
import { wordT } from "../lyrics.js";
import { compile } from "../khdl.js";

const MODULES = [
  "top", "axi_gearbox", "axi_gearbox_v2", "axi_stream", "cdc_sync_ff", "cdc_async_reset", "convcode", "encode",
  "pipeline", "external_modules", "std.math", "std.types", "std.util", "child", "delay", "Option(T)",
  "axi_passthrough", "ram_sp", "fifo", "uart_tx", "uart_rx", "crc32", "lfsr", "debounce",
  "gray_counter", "arbiter", "skid_buffer", "pwm", "spi_master", "i2c_ctrl", "rs_encoder", "viterbi",
];

const EDIT_SRC = `module top ports(
    clk: in clock,
    a: in sync(clk) uint(0..8),
    y: out sync(clk) uint(0..8),
) {
    comb {
        y = a + 1;
    }
}`;

const PY_SRC = `top = khdl.Source.new_from_manifest_path("khdl.toml").compile().resolve("top.top")
sim = top.as_verilated(build_dir).instance()
sim.ports.x.value = 4; sim.step(1)
assert sim.ports.y.value == 4`;

export const verse4 = {
  a: 229.0,
  b: 258.5,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 14 });
    const tSave = wordT("Hit save", "save");
    const tCore = wordT("Hit save", "core");
    const tSquig = wordT("The squiggles");
    const tHit = wordT("The squiggles", "hit");
    const tElab = wordT("Elaborated");
    const tTc = wordT("Elaborated", "type");
    const tVw = wordT("Elaborated", "verilog");
    const tDone = wordT("Elaborated", "done");
    const tLic = wordT("Before the license");
    const tNone = wordT("Before the license", "none");
    const tPy = wordT("Four lines of Python");
    const tPytest = wordT("Four lines of Python", "pytest");
    const tTests = wordT("Three hundred eighty");
    const tSecs = wordT("Three hundred eighty", "seconds");
    const tRust = wordT("Rust-style errors");
    const tCaret = wordT("Rust-style errors", "caret");
    const tCrime = wordT("Rust-style errors", "crime");
    const tLate = wordT("Your linter's");
    const tTime = wordT("Your linter's", "time");

    // ---- cores light up ----
    {
      const a = win(t, 229.0, tSquig - 0.05, 0.3, 0.25);
      if (a > 0) {
        const cols = 8;
        const rows = 4;
        const cw = 200;
        const ch = 120;
        const gx = W / 2 - (cols * cw + (cols - 1) * 14) / 2;
        const gy = 200;
        text(ctx, "$ khdl build -j 32", gx, 150, { size: 30, fam: F.mono, weight: 700, color: C.text, alpha: a });
        for (let i = 0; i < 32; i++) {
          const c = i % cols;
          const r = Math.floor(i / cols);
          const x = gx + c * (cw + 14);
          const y = gy + r * (ch + 14);
          const d = Math.hypot(c - 3.5, r - 1.5);
          const on = clamp((t - tSave - d * 0.06) / 0.15);
          const load = on * (0.55 + 0.45 * Math.abs(Math.sin(t * (3 + rnd(i, 1) * 5) + i)));
          rect(ctx, x, y, cw, ch, "#111622", a);
          rect(ctx, x, y + ch * (1 - load), cw, ch * load, hexA(C.cyan, 0.12 + 0.3 * load), a);
          ctx.save();
          ctx.globalAlpha = a;
          ctx.strokeStyle = on > 0 ? hexA(C.cyan, 0.3 + 0.6 * load) : C.rule;
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, cw, ch);
          ctx.restore();
          text(ctx, `core ${i}`, x + 12, y + 26, { size: 15, fam: F.mono, color: C.muted, alpha: a });
          const mod = MODULES[(i * 7 + Math.floor(Math.max(0, t - tCore) * 6)) % MODULES.length];
          if (on > 0) text(ctx, mod, x + 12, y + ch - 18, { size: 19, fam: F.mono, weight: 700, color: C.text, alpha: a * on });
        }
        const pe = prog(t, tCore + 0.8, 0.3);
        text(ctx, "every item and module elaborated in parallel", W / 2, 820, { size: 28, fam: F.sans, weight: 600, color: C.muted, align: "center", alpha: a * pe });
      }
    }

    // ---- live squiggles from the real compiler ----
    {
      const a = win(t, tSquig - 0.15, tElab - 0.02, 0.25, 0.2);
      if (a > 0) {
        const X = 260;
        const Y = 110;
        panel(ctx, X, Y, 1400, 700, { title: "top.kh — editor", alpha: a });
        const n = Math.floor(clamp((t - tSquig + 0.25) * 260, 0, EDIT_SRC.length));
        const cur = EDIT_SRC.slice(0, n);
        const size = 30;
        const res = drawCode(ctx, cur, X + 60, Y + 100, { size, alpha: a, chars: n, cursor: n < EDIT_SRC.length });
        // line numbers
        for (let l = 0; l < 9; l++) text(ctx, String(l + 1), X + 36, Y + 100 + l * size * 1.45, { size: 22, fam: F.mono, color: C.dim, align: "right", alpha: a });
        if (n >= EDIT_SRC.length) {
          const r = compile(EDIT_SRC);
          const plain = ansiPlain(r.diags);
          const m = plain.match(/--> [^:]+:(\d+):(\d+)/);
          if (m) {
            const L = parseInt(m[1]) - 1;
            const Ccol = parseInt(m[2]) - 1;
            const carets = (plain.split("\n").find((l) => /^\s*\|\s*[-\s]*\^+/.test(l)) || "^^^^^").match(/\^+/)[0].length;
            const cw = res.cw;
            const sx = X + 60 + Ccol * cw;
            const sy = Y + 100 + L * size * 1.45 + 8;
            const sp = prog(t, tHit - 0.05, 0.25);
            ctx.save();
            ctx.globalAlpha = a * sp;
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 3;
            ctx.beginPath();
            for (let k = 0; k <= carets * cw * sp; k += 3) {
              const yy = sy + Math.sin(k / 3) * 3;
              k ? ctx.lineTo(sx + k, yy) : ctx.moveTo(sx, yy);
            }
            ctx.stroke();
            ctx.restore();
            // hover popup with the real message
            const pp = prog(t, tHit + 0.35, 0.25, ease.outBack);
            if (pp > 0) {
              const first = plain.split("\n")[0];
              const fit = (plain.match(/source value with type `[^`]+` does not fit/) || [""])[0];
              ctx.save();
              ctx.translate(sx, sy + 30);
              ctx.scale(pp, pp);
              panel(ctx, 0, 0, 760, 110, { alpha: a, fill: "#1a1f2c" });
              text(ctx, first, 24, 44, { size: 24, fam: F.mono, weight: 700, color: C.red, alpha: a });
              text(ctx, fit, 24, 84, { size: 21, fam: F.mono, color: C.text, alpha: a });
              ctx.restore();
            }
            const pt = prog(t, tHit + 0.8, 0.3);
            text(ctx, `real compiler, running in this frame (WebAssembly): ${r.ms < 1 ? "<1" : r.ms.toFixed(0)} ms`, X + 1360, Y + 670, { size: 20, fam: F.mono, color: C.muted, align: "right", alpha: a * pt });
          }
        }
      }
    }

    // ---- elaborated / type-checked / verilog written, before the license server answers ----
    {
      const a = win(t, tElab - 0.15, tPy - 0.1, 0.25, 0.25);
      if (a > 0) {
        const stamps = [
          [tElab, "ELABORATED"],
          [tTc, "TYPE-CHECKED"],
          [tVw, "VERILOG WRITTEN"],
        ];
        stamps.forEach(([tt, s], i) => {
          const p = prog(t, tt - 0.05, 0.2, ease.outExpo);
          if (p <= 0) return;
          ctx.save();
          ctx.translate(560, 260 + i * 130);
          ctx.scale(lerp(1.8, 1, p), lerp(1.8, 1, p));
          text(ctx, "✓ " + s, 0, 0, { size: 64, fam: F.disp, color: C.green, align: "center", alpha: a * p });
          ctx.restore();
        });
        const pd = prog(t, tDone - 0.05, 0.25, ease.outBack);
        if (pd > 0) {
          ctx.save();
          ctx.translate(560, 700);
          ctx.scale(pd, pd);
          text(ctx, "20 ms", 0, 0, { size: 140, fam: F.disp, color: C.text, align: "center", alpha: a });
          ctx.restore();
        }
        // license server
        const pl = prog(t, tLic - 0.1, 0.3);
        if (pl > 0) {
          const X = 1060;
          const Y = 200;
          panel(ctx, X, Y, 760, 520, { title: "vendor-sim", alpha: a * pl });
          const sp = "|/-\\"[Math.floor(t * 10) % 4];
          const lines = [
            "Checking out license feature SIM_FULL ...",
            `  contacting 27000@lic-server-02 ${sp}`,
            "  queue position: 7",
            "  seats available: 0 / 40",
            "",
            "  (retrying in 300 s)",
          ];
          lines.forEach((l, k) => text(ctx, l, X + 30, Y + 90 + k * 40, { size: 24, fam: F.mono, color: k === 3 ? C.amber : C.muted, alpha: a * pl }));
          const pn = prog(t, tNone - 0.05, 0.25, ease.outBack);
          if (pn > 0) {
            ctx.save();
            ctx.translate(X + 380, Y + 420);
            ctx.rotate(-0.08);
            ctx.scale(lerp(1.6, 1, pn), lerp(1.6, 1, pn));
            ctx.globalAlpha = a * pn;
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 6;
            ctx.strokeRect(-230, -55, 460, 110);
            text(ctx, "0 SEATS", 0, 26, { size: 76, fam: F.disp, color: C.red, align: "center" });
            ctx.restore();
          }
        }
      }
    }

    // ---- four lines of python, pytest ----
    {
      const a = win(t, tPy - 0.15, tRust - 0.1, 0.25, 0.25);
      if (a > 0) {
        const X = 120;
        const Y = 110;
        panel(ctx, X, Y, 1680, 230, { title: "test_top.py", alpha: a });
        const n = Math.floor(clamp((t - tPy + 0.1) * 150, 0, PY_SRC.length));
        drawCode(ctx, PY_SRC, X + 40, Y + 90, { size: 22, lang: "verilog", alpha: a, chars: n, cursor: n < PY_SRC.length });
        const pt = prog(t, tPytest - 0.15, 0.3);
        if (pt > 0) {
          const TX = 120;
          const TY = 380;
          panel(ctx, TX, TY, 1680, 440, { title: "terminal", alpha: a * pt });
          text(ctx, "$ uv run pytest -n auto", TX + 40, TY + 80, { size: 26, fam: F.mono, weight: 700, color: C.text, alpha: a * pt });
          const total = 428;
          const dp = clamp((t - tPytest) / (tSecs - tPytest + 0.1));
          const nd = Math.floor(total * ease.inOutQuad(dp));
          const perLine = 72;
          for (let k = 0; k < Math.ceil(nd / perLine); k++) {
            const cnt = Math.min(perLine, nd - k * perLine);
            let s = "";
            for (let j = 0; j < cnt; j++) {
              const id = k * perLine + j;
              s += id === 251 ? "s" : id === 397 ? "x" : ".";
            }
            text(ctx, s, TX + 40, TY + 130 + k * 34, { size: 24, fam: F.mono, color: C.green, alpha: a * pt });
            text(ctx, `[${String(Math.floor(((k * perLine + cnt) / total) * 100)).padStart(3)}%]`, TX + 1500, TY + 130 + k * 34, { size: 24, fam: F.mono, color: C.muted, alpha: a * pt });
          }
          const ps = prog(t, tSecs - 0.1, 0.25, ease.outBack);
          if (ps > 0) {
            ctx.save();
            ctx.translate(W / 2, TY + 370);
            ctx.scale(ps, ps);
            glow(ctx, C.green, 30, () => text(ctx, window.COMPILER.pytest, 0, 0, { size: 46, fam: F.mono, weight: 800, color: C.green, align: "center", alpha: a }));
            ctx.restore();
            text(ctx, "(the lyrics were written a few hundred tests ago)", W / 2, TY + 420, { size: 22, fam: F.sans, color: C.muted, align: "center", alpha: a * prog(t, tSecs + 0.6, 0.4) });
          }
        }
      }
    }

    // ---- rust-style caret; linter late, compiler on time ----
    {
      const a = win(t, tRust - 0.15, 258.5, 0.25, 0.3);
      if (a > 0) {
        const zoom = prog(t, tCaret - 0.2, 0.8, ease.inOutCubic);
        const out = prog(t, tLate - 0.1, 0.4, ease.inOutCubic);
        ctx.save();
        ctx.globalAlpha = a * (1 - out);
        ctx.translate(W / 2, 470);
        const sc = lerp(1, 1.7, zoom);
        ctx.scale(sc, sc);
        panel(ctx, -560, -190, 1120, 330, { title: "khdl build", alpha: 1 });
        drawAnsi(ctx, window.COMPILER.diags.trunc, -520, -110, { size: 21 });
        ctx.restore();
        const pc = prog(t, tCrime - 0.05, 0.25, ease.outBack);
        if (pc > 0 && out < 1) {
          ctx.save();
          ctx.translate(1330, 800);
          ctx.rotate(-0.06);
          ctx.scale(pc, pc);
          ctx.globalAlpha = a * (1 - out);
          ctx.fillStyle = "#ffd65a";
          ctx.fillRect(-170, -40, 340, 80);
          text(ctx, "CRIME SCENE", 0, 16, { size: 44, fam: F.disp, color: "#1a1400", align: "center" });
          ctx.restore();
        }
        if (out > 0) {
          // calendar vs stopwatch
          ctx.save();
          ctx.globalAlpha = a * out;
          panel(ctx, 260, 250, 560, 460, { title: "linter", alpha: 1 });
          const months = ["JUL", "AUG", "SEP", "OCT"];
          const mi = Math.min(3, Math.floor(clamp((t - tLate) / 1.6) * 4));
          text(ctx, months[mi], 540, 440, { size: 120, fam: F.disp, color: C.muted, align: "center" });
          text(ctx, "report ready: +3 months", 540, 560, { size: 28, fam: F.mono, color: C.amber, align: "center" });
          panel(ctx, 1100, 250, 560, 460, { title: "khdl", alpha: 1 });
          glow(ctx, C.green, 30, () => text(ctx, "20 ms", 1380, 450, { size: 130, fam: F.disp, color: C.green, align: "center" }));
          text(ctx, "every save", 1380, 560, { size: 28, fam: F.mono, color: C.muted, align: "center" });
          ctx.restore();
          const pt = prog(t, tTime - 0.05, 0.25, ease.outBack);
          if (pt > 0) {
            ctx.save();
            ctx.translate(1380, 650);
            ctx.scale(pt, pt);
            text(ctx, "on time ✓", 0, 0, { size: 44, fam: F.disp, color: C.green, align: "center", alpha: a });
            ctx.restore();
          }
        }
      }
    }
  },
};

// ---------- pre-chorus 3: Verilog 2005 out, into every tool ----------
const TOOLS = ["simulator", "synthesis", "lint", "equivalence", "FPGA flow", "place & route", "formal", "power"];

export const pre3 = {
  a: 258.2,
  b: 274.6,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 14 });
    const t2005 = wordT("It all comes out", "two");
    const tTool = wordT("Every tool you've");
    const tAlive = wordT("Every tool you've", "alive");
    const tWait = wordT("No waiting ten");
    const tStruct = wordT("No waiting ten", "struct");
    const tLives = wordT("The language lives");
    const tUnstuck = wordT("The language lives", "unstuck");
    const a = win(t, 258.2, 274.6, 0.3, 0.4);
    // source -> verilog streaming
    {
      const al = a * (1 - prog(t, tWait - 0.2, 0.4));
      if (al > 0) {
        const src = window.COMPILER.readme_src;
        const vl = window.COMPILER.readme_verilog;
        panel(ctx, 70, 90, 760, 690, { title: "top.kh", alpha: al });
        drawCode(ctx, src, 100, 170, { size: 17.5, alpha: al });
        panel(ctx, 1090, 90, 760, 690, { title: "top.v — Verilog 2005", alpha: al });
        const scroll = Math.max(0, t - 259) * 260;
        ctx.save();
        ctx.beginPath();
        ctx.rect(1090, 130, 760, 645);
        ctx.clip();
        drawCode(ctx, vl, 1120, 170 - scroll % 1500, { size: 16, lang: "verilog", alpha: al });
        ctx.restore();
        // flowing arrow between them
        for (let k = 0; k < 6; k++) {
          const ph = (t * 1.6 + k / 6) % 1;
          ctx.save();
          ctx.globalAlpha = al * Math.sin(ph * Math.PI);
          ctx.fillStyle = C.cyan;
          ctx.beginPath();
          const x = lerp(850, 1070, ph);
          ctx.moveTo(x, 440);
          ctx.lineTo(x - 18, 425);
          ctx.lineTo(x - 18, 455);
          ctx.fill();
          ctx.restore();
        }
        const p5 = prog(t, t2005 - 0.1, 0.3, ease.outBack);
        if (p5 > 0) {
          ctx.save();
          ctx.translate(960, 560);
          ctx.scale(p5, p5);
          text(ctx, "2005", 0, 0, { size: 60, fam: F.disp, color: C.text, align: "center", alpha: al });
          text(ctx, "plain Verilog", 0, 36, { size: 20, fam: F.mono, color: C.muted, align: "center", alpha: al });
          ctx.restore();
        }
        // every tool lights up
        const pt = prog(t, tTool - 0.2, 0.3);
        if (pt > 0) {
          TOOLS.forEach((name, i) => {
            const x = 140 + i * 210;
            const y = 815;
            const on = prog(t, tTool + i * 0.28, 0.2);
            rect(ctx, x, y, 190, 60, on > 0 ? hexA(C.green, 0.15 * on) : "#121620", al * pt);
            ctx.save();
            ctx.globalAlpha = al * pt;
            ctx.strokeStyle = on > 0 ? C.green : C.rule;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, 190, 60);
            ctx.restore();
            text(ctx, (on >= 1 ? "✓ " : "") + name, x + 95, y + 38, { size: 20, fam: F.mono, weight: 700, color: on >= 1 ? C.green : C.muted, align: "center", alpha: al * pt });
          });
        }
      }
    }
    // the ten-year support ticket
    {
      const al = a * win(t, tWait - 0.2, tLives - 0.05, 0.35, 0.3);
      if (al > 0) {
        panel(ctx, 360, 140, 1200, 560, { title: "vendor support portal", alpha: al });
        text(ctx, "Ticket #48213", 420, 240, { size: 44, fam: F.disp, color: C.text, alpha: al });
        text(ctx, "SystemVerilog: packed struct in module port not supported", 420, 300, { size: 28, fam: F.sans, weight: 600, color: C.muted, alpha: al });
        const yr = Math.floor(lerp(2016, 2026, clamp((t - tWait) / (tStruct - tWait))));
        text(ctx, `opened 2016  ·  status: OPEN  ·  ${yr}`, 420, 380, { size: 30, fam: F.mono, weight: 700, color: C.amber, alpha: al });
        const replies = ["“We are tracking this for a future release.”", "“Workaround: flatten the struct.”", "“Escalated to R&D.”", "“Fixed in 2027.1 (beta).”"];
        replies.forEach((r, k) => {
          const pr = prog(t, tWait + 0.3 + k * 0.6, 0.3);
          text(ctx, r, 440, 450 + k * 48, { size: 24, fam: F.sans, weight: 500, color: C.muted, alpha: al * pr });
        });
        const ps = prog(t, tStruct - 0.05, 0.2, ease.outExpo);
        if (ps > 0) {
          ctx.save();
          ctx.translate(1270, 560);
          ctx.rotate(-0.18);
          ctx.scale(lerp(2, 1, ps), lerp(2, 1, ps));
          ctx.globalAlpha = al * ps;
          ctx.strokeStyle = C.green;
          ctx.lineWidth = 7;
          ctx.strokeRect(-200, -55, 400, 110);
          text(ctx, "NOT NEEDED", 0, 20, { size: 56, fam: F.disp, color: C.green, align: "center" });
          ctx.restore();
        }
      }
    }
    // the language lives in the compiler
    {
      const al = a * win(t, tLives - 0.15, 274.6, 0.3, 0.4);
      if (al > 0) {
        const cx = W / 2;
        const cy = 440;
        panel(ctx, cx - 240, cy - 120, 480, 240, { alpha: al, fill: "#101826" });
        text(ctx, "khdl", cx, cy + 30, { size: 96, fam: F.disp, color: C.text, align: "center", alpha: al });
        const feats = ["range types", "domains", "interfaces", "generics", "enums", "packages"];
        feats.forEach((f, i) => {
          const ang = (i / feats.length) * Math.PI * 2 + t * 0.4;
          const r = 380;
          const p = prog(t, tLives + i * 0.12, 0.3, ease.outBack);
          text(ctx, f, cx + Math.cos(ang) * r, cy + Math.sin(ang) * r * 0.5 + 10, { size: 28, fam: F.mono, weight: 700, color: [C.cyan, C.magenta, C.amber, C.green, C.violet, C.cyan][i], align: "center", alpha: al * p });
        });
        text(ctx, "→ Verilog 2005 → the tools you already have", cx, cy + 330, { size: 28, fam: F.sans, weight: 600, color: C.muted, align: "center", alpha: al });
        const pu = prog(t, tUnstuck - 0.1, 0.3);
        if (pu > 0) {
          const f = Math.exp(-Math.max(0, t - tUnstuck) * 3);
          rect(ctx, 0, 0, W, H, C.green, 0.12 * f * al);
        }
      }
    }
  },
};

// ---------- key change: the real playground ----------
const IMGS = { initial: null, fixed: null, frames: [] };

export const keychange = {
  a: 274.0,
  b: 284.2,
  lyrics: "none",
  async init() {
    const load = (src) =>
      new Promise((res) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = () => res(null);
        im.src = src;
      });
    IMGS.initial = await load("assets/playground/initial.png");
    IMGS.fixed = await load("assets/playground/fixed.png");
    const n = window.COMPILER.pg_frames;
    IMGS.frames = await Promise.all(Array.from({ length: n }, (_, i) => load(`assets/playground/type_${String(i).padStart(4, "0")}.png`)));
  },
  render(ctx, t) {
    const a = win(t, 274.0, 284.2, 0.4, 0.35);
    dotGrid(ctx, t, { alpha: 0.5 * a, ox: -t * 30 });
    const k = kick(t);
    // which image
    const t0 = 274.2;
    const tType = 275.6;
    const tEnd = 280.6;
    const tFix = 282.2;
    let im = IMGS.initial;
    if (t >= tType && t < tFix) {
      const n = IMGS.frames.length;
      im = IMGS.frames[Math.min(n - 1, Math.floor(((t - tType) / (tEnd - tType)) * n))];
    } else if (t >= tFix) im = IMGS.fixed;
    // browser chrome
    const BW = 1500;
    const BH = 844 + 50;
    const zoomIn = 1 - ease.outCubic(clamp((t - t0) / 0.8));
    const sc = 1 + 0.012 * k - zoomIn * 0.15;
    ctx.save();
    ctx.translate(W / 2, H / 2 + 20);
    ctx.scale(sc, sc);
    ctx.rotate(lerp(-0.04, 0, ease.outCubic(clamp((t - t0) / 1.2))));
    const x = -BW / 2;
    const y = -BH / 2;
    panel(ctx, x, y, BW, BH, { alpha: a, fill: "#1b1f27", radius: 14 });
    ctx.globalAlpha = a;
    rect(ctx, x + 120, y + 10, BW - 240, 32, "#2a2f3a", a);
    text(ctx, "🔒  khdl-lang.org", x + 140, y + 33, { size: 18, fam: F.sans, weight: 500, color: "#c0c6d2", alpha: a });
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = [C.red, C.amber, C.green][i];
      ctx.beginPath();
      ctx.arc(x + 26 + i * 22, y + 26, 7, 0, Math.PI * 2);
      ctx.fill();
    }
    if (im) ctx.drawImage(im, x, y + 50, BW, BW * (900 / 1600));
    ctx.restore();
    // captions
    const caps = [
      [t0 + 0.3, tType, "the real compiler, compiled to WebAssembly"],
      [tType + 0.2, tEnd + 0.8, "type — it checks as you go"],
      [tEnd + 0.6, tFix, "invalid domain crossing: different clock"],
      [tFix, 284.2, "fix the domain → Verilog, instantly"],
    ];
    caps.forEach(([s, e, c], i) => {
      const ca = win(t, s, e, 0.2, 0.2) * a;
      if (ca <= 0) return;
      rect(ctx, W / 2 - 520, 990, 1040, 56, C.bg, ca * 0.85);
      text(ctx, c, W / 2, 1028, { size: 32, fam: i === 2 ? F.mono : F.sans, weight: 700, color: i === 2 ? C.red : C.text, align: "center", alpha: ca });
    });
  },
};

// ---------- outro ----------
export const outro = {
  a: 310.2,
  b: 331.3,
  lyrics: "none",
  render(ctx, t) {
    const tBuild = wordT("K H D L build");
    const tFin = wordT("K H D L build", "twenty");
    const tZero = wordT("Zero errors");
    const tWarn = wordT("Zero errors", "warnings");
    const tIf = wordT("If it compiles... it fits.");
    const tFits = wordT("If it compiles... it fits.", "fits");
    const a = win(t, 310.2, 331.3, 0.5, 0.01);
    dotGrid(ctx, t, { alpha: 0.4 * a * (1 - prog(t, 324, 4)), ox: -t * 6 });
    // terminal
    const ta = a * (1 - prog(t, tIf + 3.2, 1.0));
    if (ta > 0) {
      const X = 360;
      const Y = 200;
      panel(ctx, X, Y, 1200, 420, { title: "~/project", alpha: ta });
      const cmd = "$ khdl build";
      const n = Math.floor(clamp((t - tBuild + 0.2) * 14) * cmd.length);
      text(ctx, cmd.slice(0, n), X + 50, Y + 110, { size: 44, fam: F.mono, weight: 700, color: C.text, alpha: ta });
      if (n < cmd.length || (t < tFin && Math.floor(t * 2.5) % 2)) rect(ctx, X + 50 + measure(ctx, cmd.slice(0, n), 44, F.mono, 700) + 6, Y + 76, 24, 44, C.text, ta);
      const pf = prog(t, tFin - 0.1, 0.2);
      text(ctx, "    Finished in 20 ms", X + 50, Y + 190, { size: 44, fam: F.mono, weight: 700, color: C.green, alpha: ta * pf });
      const pz = prog(t, tZero - 0.05, 0.2);
      text(ctx, "    0 errors", X + 50, Y + 270, { size: 44, fam: F.mono, weight: 700, color: C.text, alpha: ta * pz });
      const pw = prog(t, tWarn - 0.05, 0.2);
      text(ctx, "    0 warnings", X + 50, Y + 340, { size: 44, fam: F.mono, weight: 700, color: C.text, alpha: ta * pw });
    }
    // if it compiles... it fits.
    const pi = prog(t, tIf - 0.05, 0.3);
    const ia = a * (1 - prog(t, tIf + 3.2, 1.0));
    if (pi > 0 && ia > 0) {
      text(ctx, "if it compiles…", W / 2, 780, { size: 76, fam: F.disp, color: C.text, align: "center", alpha: ia * pi });
      const pf = prog(t, tFits - 0.05, 0.3, ease.outBack);
      if (pf > 0) {
        ctx.save();
        ctx.translate(W / 2, 890);
        ctx.scale(pf, pf);
        glow(ctx, C.green, 40, () => text(ctx, "it fits.", 0, 0, { size: 76, fam: F.disp, color: C.green, align: "center", alpha: ia }));
        ctx.restore();
      }
    }
    // end card
    const ea = a * win(t, tIf + 3.6, 331.3, 0.8, 1.6);
    if (ea > 0) {
      glow(ctx, C.cyan, 40, () => text(ctx, "KHDL", W / 2, 470, { size: 240, fam: F.disp, color: C.text, align: "center", alpha: ea, spacing: 16 }));
      text(ctx, "a hardware description language that checks what you meant", W / 2, 560, { size: 32, fam: F.sans, weight: 500, color: C.muted, align: "center", alpha: ea });
      text(ctx, "khdl-lang.org   ·   github.com/khdl-lang/khdl", W / 2, 640, { size: 30, fam: F.mono, weight: 700, color: C.cyan, align: "center", alpha: ea });
      // final clock: ticks, then stops
      const stop = 329.0;
      const b = A.data.beats;
      const sig = (tau) => {
        if (tau > stop) return 0;
        const { phase } = beatInfo(tau);
        return phase < 0.5 ? 1 : 0;
      };
      const headX = W * 0.8;
      const x0 = W * 0.2;
      const span = 4;
      ctx.save();
      ctx.globalAlpha = ea;
      ctx.strokeStyle = C.cyan;
      ctx.lineWidth = 3;
      ctx.shadowColor = C.cyan;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      let prev = null;
      for (let i = 0; i <= 600; i++) {
        const x = x0 + ((headX - x0) * i) / 600;
        const tau = t - ((headX - x) / (headX - x0)) * span;
        const yy = 800 - sig(tau) * 50;
        if (prev === null) ctx.moveTo(x, yy);
        else if (yy !== prev) ctx.lineTo(x, prev), ctx.lineTo(x, yy);
        else ctx.lineTo(x, yy);
        prev = yy;
      }
      ctx.stroke();
      ctx.restore();
    }
  },
};
