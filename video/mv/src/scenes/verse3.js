import { C, W, H, clamp, ease, prog, win, pulse, hexA, lerp, rnd, kick } from "../lib.js";
import { F, text, measure, panel, dotGrid, line, glow, rect, drawAnsi, drawCode } from "../draw.js";
import { wordT } from "../lyrics.js";
import { compile } from "../khdl.js";

const AXI_IFACE = `interface axi_stream(T: type) {
    data: T,
    valid: bool,
    ready: bool,

    view input  { data: in,  valid: in,  ready: out }
    view output { data: out, valid: out, ready: in  }
}`;

function block(ctx, x, y, w, h, title, col, alpha) {
  panel(ctx, x, y, w, h, { alpha, fill: "#121722" });
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = col;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
  text(ctx, title, x + w / 2, y + 44, { size: 30, fam: F.mono, weight: 800, color: col, align: "center", alpha });
}

const NWIRES = 44;

export const verse3 = {
  a: 168.5,
  b: 197.3,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 8 });
    const t0 = wordT("Remember wiring");
    const tPorts = wordT("Remember wiring", "ports");
    const tDot = wordT("Dot valid");
    const tSwap = wordT("Dot valid", "swapped");
    const tLint = wordT("Dot valid", "lint");
    const tIface = wordT("Now I write the interface");
    const tView = wordT("Now I write the interface", "view");
    const tIn = wordT("Input here");
    const tChecks = wordT("Input here", "checks");
    const tStream = wordT("Every stream is one port");
    const tTop = wordT("The top level reads");
    const tDiagram = wordT("The top level reads", "block");
    const tHand = wordT("The handshake lives");
    const tMis = wordT("The handshake lives", "misuse");
    const tReview = wordT("So the review");
    const tGlue = wordT("So the review", "thousand");

    // ---- A/B: spaghetti port list collapsing into one interface ----
    {
      const a = win(t, t0 - 0.3, tStream - 0.1, 0.35, 0.35);
      if (a > 0) {
        const LX = 160;
        const RX = 1460;
        const BW = 300;
        const BY = 150;
        const collapse = prog(t, tIface, 1.0, ease.inOutCubic);
        const BH = lerp(640, 300, collapse);
        block(ctx, LX, BY, BW, BH, "u_dma", C.cyan, a);
        block(ctx, RX, BY, BW, BH, "u_fifo", C.magenta, a);
        const nShown = Math.floor(clamp((t - t0 + 0.2) / (tDot - t0)) * NWIRES);
        const swap = prog(t, tSwap - 0.1, 0.4, ease.inOutCubic);
        const vI = 20;
        const rI = 21;
        for (let i = 0; i < NWIRES; i++) {
          if (i >= nShown) break;
          const y0 = BY + 90 + i * ((BH - 120) / NWIRES);
          let y1 = y0;
          if (i === vI) y1 = lerp(y0, BY + 90 + rI * ((BH - 120) / NWIRES), swap);
          if (i === rI) y1 = lerp(y0, BY + 90 + vI * ((BH - 120) / NWIRES), swap);
          const yc = BY + BH / 2;
          const ya = lerp(y0, yc, collapse);
          const yb = lerp(y1, yc, collapse);
          const special = i === vI || i === rI;
          const hiOn = t > tDot - 0.1;
          const col = special && hiOn ? (i === vI ? C.amber : C.cyan) : hexA(C.text, 0.25 + 0.15 * rnd(i, 4));
          ctx.save();
          ctx.globalAlpha = a * (special ? 1 : 1 - collapse * 0.7);
          ctx.strokeStyle = col;
          ctx.lineWidth = special && hiOn ? 4 : 1.5;
          ctx.beginPath();
          ctx.moveTo(LX + BW, ya);
          ctx.bezierCurveTo(LX + BW + 400, ya, RX - 400, yb, RX, yb);
          ctx.stroke();
          ctx.restore();
        }
        if (t > tDot - 0.1 && collapse < 0.5) {
          const yv = BY + 90 + vI * ((BH - 120) / NWIRES);
          const yr = BY + 90 + rI * ((BH - 120) / NWIRES);
          text(ctx, ".valid(valid)", LX + BW + 20, yv - 18, { size: 22, fam: F.mono, weight: 700, color: C.amber, alpha: a * (1 - collapse * 2) });
          text(ctx, ".ready(ready)", LX + BW + 20, yr + 34, { size: 22, fam: F.mono, weight: 700, color: C.cyan, alpha: a * (1 - collapse * 2) });
        }
        // port counter
        const nPorts = Math.min(200, Math.floor(clamp((t - t0 + 0.2) / (tDot - t0)) * 200));
        if (collapse < 1) text(ctx, `${nPorts} ports`, W / 2, 120, { size: 40, fam: F.disp, color: C.text, align: "center", alpha: a * (1 - collapse) });
        // lint says fine
        const pl = prog(t, tLint - 0.1, 0.3, ease.outBack);
        if (pl > 0 && collapse < 1) {
          ctx.save();
          ctx.translate(W / 2, 860);
          ctx.scale(pl, pl);
          panel(ctx, -240, -40, 480, 80, { alpha: a * (1 - collapse) });
          text(ctx, "lint: 0 warnings ✓", 0, 12, { size: 32, fam: F.mono, weight: 800, color: C.green, align: "center", alpha: a * (1 - collapse) });
          ctx.restore();
        }
        // the bus
        if (collapse > 0) {
          const yc = BY + BH / 2;
          ctx.save();
          ctx.globalAlpha = a * collapse;
          ctx.strokeStyle = C.green;
          ctx.lineWidth = 26 * collapse;
          ctx.shadowColor = C.green;
          ctx.shadowBlur = 30;
          ctx.beginPath();
          ctx.moveTo(LX + BW, yc);
          ctx.lineTo(RX, yc);
          ctx.stroke();
          ctx.restore();
          text(ctx, "axi_stream(uint(8))", W / 2, yc - 36, { size: 30, fam: F.mono, weight: 800, color: C.green, align: "center", alpha: a * collapse });
          const pv = prog(t, tView - 0.1, 0.3, ease.outBack);
          if (pv > 0) {
            text(ctx, ".output", LX + BW + 20, yc + 60, { size: 30, fam: F.mono, weight: 800, color: C.cyan, alpha: a * pv });
            text(ctx, ".input", RX - 20, yc + 60, { size: 30, fam: F.mono, weight: 800, color: C.magenta, align: "right", alpha: a * pv });
          }
          const pc = prog(t, tChecks - 0.1, 0.3, ease.outBack);
          if (pc > 0) {
            ctx.save();
            ctx.translate(W / 2, yc + 70);
            ctx.scale(pc, pc);
            text(ctx, "directions checked ✓", 0, 0, { size: 28, fam: F.mono, weight: 700, color: C.green, align: "center", alpha: a });
            ctx.restore();
          }
          // the real interface definition
          const pi = prog(t, tIface + 0.6, 0.4);
          panel(ctx, 460, 540, 1000, 330, { title: "axi_stream.kh", alpha: a * pi });
          drawCode(ctx, AXI_IFACE, 500, 620, { size: 22, alpha: a * pi, chars: Math.floor((t - tIface - 0.6) * 160) });
        }
      }
    }

    // ---- C: the top level reads like the block diagram ----
    {
      const a = win(t, tStream - 0.2, tHand - 0.1, 0.35, 0.3);
      if (a > 0) {
        const src = window.COMPILER.readme_src.split("\n").slice(0, 11).join("\n");
        panel(ctx, 60, 110, 860, 560, { title: "top.kh", alpha: a });
        const hi = (ln) => (ln >= 5 && ln <= 9 ? 1 : 0.45);
        drawCode(ctx, src, 90, 190, { size: 21, alpha: a, lineAlpha: (ln) => (t > tStream + 0.3 ? hi(ln) : 0.9) });
        // diagram draws itself
        const dp = prog(t, tStream + 0.2, 1.4, ease.inOutCubic);
        const BX = 1290;
        const BY = 200;
        const BW = 300;
        const BH = 440;
        ctx.save();
        ctx.globalAlpha = a;
        ctx.strokeStyle = C.text;
        ctx.lineWidth = 3;
        const per = 2 * (BW + BH);
        ctx.setLineDash([per * dp, per]);
        ctx.strokeRect(BX, BY, BW, BH);
        ctx.setLineDash([]);
        ctx.restore();
        text(ctx, "top", BX + BW / 2, BY + 50, { size: 36, fam: F.mono, weight: 800, color: C.text, align: "center", alpha: a * dp });
        const ports = [
          ["clk", true, BY + 110, C.cyan, 4],
          ["data_to_add", true, BY + 190, C.text, 4],
          ["stream_in", true, BY + 300, C.green, 18],
          ["stream_out", false, BY + 300, C.green, 18],
          ["values_transferred", false, BY + 190, C.text, 4],
        ];
        ports.forEach(([name, isIn, y, col, lw], i) => {
          const pp = prog(t, tStream + 1.0 + i * 0.35, 0.35);
          if (pp <= 0) return;
          const x0 = isIn ? BX - 280 : BX + BW;
          const x1 = isIn ? BX : BX + BW + 280;
          ctx.save();
          ctx.globalAlpha = a * pp;
          ctx.strokeStyle = col;
          ctx.lineWidth = lw;
          ctx.beginPath();
          ctx.moveTo(isIn ? x0 : x0, y);
          ctx.lineTo(isIn ? lerp(x0, x1, pp) : lerp(x0, x1, pp), y);
          ctx.stroke();
          ctx.restore();
          text(ctx, name, isIn ? x0 : x1, y - 16, { size: 20, fam: F.mono, weight: 700, color: col, align: isIn ? "left" : "right", alpha: a * pp });
        });
        const ps = prog(t, tStream + 0.4, 0.4);
        text(ctx, "one port = one stream: its clock, its reset, its type", 1440, 740, { size: 24, fam: F.sans, weight: 600, color: C.muted, align: "center", alpha: a * ps });
      }
    }

    // ---- D: handshake behind an API; review the design, not the glue ----
    {
      const a = win(t, tHand - 0.2, 197.4, 0.3, 0.25);
      if (a > 0) {
        // the bus with a lock on the handshake
        const yb = 320;
        ctx.save();
        ctx.globalAlpha = a;
        ctx.strokeStyle = C.green;
        ctx.lineWidth = 26;
        ctx.shadowColor = C.green;
        ctx.shadowBlur = 30;
        ctx.beginPath();
        ctx.moveTo(200, yb);
        ctx.lineTo(1720, yb);
        ctx.stroke();
        ctx.restore();
        const pl = prog(t, tHand + 0.2, 0.4, ease.outBack);
        if (pl > 0) {
          ctx.save();
          ctx.translate(W / 2, yb);
          ctx.scale(pl, pl);
          rect(ctx, -60, -40, 120, 90, C.amber, a);
          ctx.strokeStyle = C.amber;
          ctx.lineWidth = 12;
          ctx.globalAlpha = a;
          ctx.beginPath();
          ctx.arc(0, -40, 38, Math.PI, 0);
          ctx.stroke();
          text(ctx, "valid · ready", 0, 90, { size: 26, fam: F.mono, weight: 800, color: C.amber, align: "center", alpha: a });
          ctx.restore();
        }
        const pm = prog(t, tMis - 0.1, 0.3);
        text(ctx, "the protocol lives in one place — everyone else just sends and receives", W / 2, 470, { size: 28, fam: F.sans, weight: 600, color: C.muted, align: "center", alpha: a * pm });
        // a thousand lines of glue, collapsing into one instance
        const pg = prog(t, tReview - 0.1, 0.3);
        if (pg > 0) {
          const crush = prog(t, tGlue, 0.6, ease.inCubic);
          const lines = [];
          const sigs = ["tdata", "tvalid", "tready", "tlast", "tkeep", "tuser"];
          for (let k = 0; k < 18; k++) lines.push(`    .m_axis_${sigs[k % 6]}_${Math.floor(k / 6)} (dma_to_fifo_${sigs[k % 6]}_${Math.floor(k / 6)}),`);
          const h = lerp(18 * 22, 0, crush);
          ctx.save();
          ctx.beginPath();
          ctx.rect(300, 540, 1320, h + 4);
          ctx.clip();
          lines.forEach((l, k) => text(ctx, l, 340, 560 + k * 22, { size: 18, fam: F.mono, color: C.muted, alpha: a * pg }));
          ctx.restore();
          if (crush > 0) {
            const s = "instance fifo ports(clk, rst, input=dma_out, output=fifo_out);";
            const pa = prog(t, tGlue + 0.55, 0.3);
            drawCode(ctx, s, W / 2 - measure(ctx, s, 30, F.mono, 400) / 2, 640, { size: 30, alpha: a * pa });
          }
          const n = Math.round(lerp(1000, 1, crush));
          text(ctx, `${n} line${n === 1 ? "" : "s"} of glue`, W / 2, 800, { size: 36, fam: F.disp, color: crush >= 1 ? C.green : C.text, align: "center", alpha: a * pg });
        }
      }
    }
  },
};

// ---------- bridge ----------
const TOOLS = [
  ["gen_regs.pl", "Perl", "#!/usr/bin/perl\nforeach my $r (@regs) {\n  print \"reg [$r->{w}-1:0] $r->{n};\\n\";\n}", "Perl"],
  ["gen_perl.py", "Python writing Perl", "with open('gen_regs.pl', 'w') as f:\n    f.write(PERL_TEMPLATE.format(\n        regs=regs))", "Python"],
  ["ports.v.j2", "Jinja", "{% for p in ports %}\n  .{{ p.name }}({{ p.net }}),\n{% endfor %}", "Jinja"],
  ["build.tcl", "Tcl", "foreach f [glob rtl/*.v] {\n  read_verilog $f\n}\nsynth_design -top top", "Tcl"],
  ["lint.cron", "nightly lint", "0 2 * * * cd ~/proj &&\n  ./run_lint.sh > lint.log\n# nobody reads lint.log", "lint"],
  ["Makefile", "held with tape", "# DO NOT TOUCH (2019)\nall: gen\n\t./gen_perl.py && perl gen_regs.pl\n\tvivado -mode batch -source build.tcl", "build"],
];

export const bridge = {
  a: 197.0,
  b: 221.0,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 20 });
    const k = kick(t);
    // ---- B1: four real diagnostics, slammed on the hooks ----
    const hooks = [
      [wordT('"Driver mismatch'), "latch"],
      [wordT('"Self-loop'), "loop"],
      [wordT('"Multiple drivers'), "multi"],
      [wordT('"Different clock'), "cdc"],
    ];
    const tPerl = wordT("No Perl");
    {
      const a = win(t, 197.0, tPerl - 0.05, 0.1, 0.3);
      if (a > 0) {
        rect(ctx, 0, 0, W, H, C.red, 0.05 * k * a);
        hooks.forEach(([tt, key], i) => {
          const p = clamp((t - tt + 0.04) / 0.16);
          if (p <= 0) return;
          const next = hooks[i + 1] ? hooks[i + 1][0] : tPerl;
          const older = t > next ? 1 : 0;
          const sc = lerp(1.25, 1, ease.outExpo(p)) * (older ? 0.94 : 1);
          ctx.save();
          ctx.translate(W / 2 + (i - 1.5) * 40, 470 + (i - 1.5) * 30);
          ctx.rotate((rnd(i, 9) - 0.5) * 0.08);
          ctx.scale(sc, sc);
          const da = a * (older ? 0.35 : 1);
          panel(ctx, -760, -330, 1520, 640, { title: "khdl build", alpha: da });
          drawAnsi(ctx, window.COMPILER.diags[key], -720, -250, { size: 21, alpha: da });
          ctx.restore();
          if (!older) {
            const f = Math.exp(-(t - tt) * 10);
            rect(ctx, 0, 0, W, H, C.red, 0.2 * f * a);
          }
        });
      }
    }
    // ---- B2: the tool graveyard ----
    const tTypes = wordT("Types are values");
    const tCompiles = wordT("Types are values", "compiles");
    {
      const a = win(t, tPerl - 0.15, tTypes + 0.3, 0.2, 0.3);
      if (a > 0) {
        const kill = [
          wordT("No Perl", "verilog"),
          wordT("No Perl", "writing") + 0.6,
          wordT("No Jinja", "lists"),
          wordT("No Jinja", "world"),
          wordT("No homegrown", "cron"),
          wordT("No homegrown", "tape"),
        ];
        TOOLS.forEach(([fname, label, body], i) => {
          const col = i % 3;
          const row = Math.floor(i / 3);
          const x = 110 + col * 580;
          const y = 120 + row * 380;
          const pin = prog(t, tPerl - 0.15 + i * 0.05, 0.3, ease.outBack);
          const dead = t - kill[i];
          const fall = dead > 0.35 ? (dead - 0.35) : 0;
          ctx.save();
          ctx.translate(x + 270, y + 160 + fall * fall * 2200);
          ctx.rotate(fall * (rnd(i, 3) - 0.5) * 3);
          ctx.scale(pin, pin);
          panel(ctx, -270, -160, 540, 320, { title: fname, alpha: a });
          drawCode(ctx, body, -240, -80, { size: 19, lang: "verilog", alpha: a * (dead > 0 ? 0.5 : 1) });
          if (i === 5) {
            // duct tape
            ctx.save();
            ctx.rotate(-0.3);
            rect(ctx, -280, -20, 560, 46, "#9aa0a6", a * 0.85);
            ctx.restore();
          }
          text(ctx, label, 0, 135, { size: 22, fam: F.sans, weight: 700, color: C.muted, align: "center", alpha: a });
          if (dead > 0) {
            const sp = clamp(dead / 0.15);
            ctx.save();
            ctx.globalAlpha = a;
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 12;
            ctx.lineCap = "round";
            ctx.shadowColor = C.red;
            ctx.shadowBlur = 20;
            ctx.beginPath();
            ctx.moveTo(-250, -140);
            ctx.lineTo(lerp(-250, 250, sp), lerp(-140, 140, sp));
            ctx.stroke();
            ctx.restore();
          }
          ctx.restore();
        });
      }
    }
    // ---- one language: types are values, modules are functions ----
    {
      const a = win(t, tTypes - 0.1, 221.0, 0.3, 0.4);
      if (a > 0) {
        const src = window.COMPILER.bridge_src;
        panel(ctx, 360, 110, 1200, 640, { title: "everything.kh", alpha: a });
        const ln = (l) => {
          const tv = wordT("Types are values", "values");
          const tm = wordT("Types are values", "modules");
          if (l >= 2 && l <= 3) return t > tv - 0.2 ? 1 : 0.35;
          if (l >= 5 && l <= 6) return t > tm - 0.2 ? 1 : 0.35;
          return 0.6;
        };
        drawCode(ctx, src, 410, 200, { size: 28, alpha: a, lineAlpha: ln });
        const pc = prog(t, tCompiles - 0.05, 0.3, ease.outBack);
        if (pc > 0) {
          const r = compile(src);
          ctx.save();
          ctx.translate(W / 2, 830);
          ctx.scale(pc, pc);
          text(ctx, "✓ compiles", 0, 0, { size: 64, fam: F.disp, color: C.green, align: "center", alpha: a });
          ctx.restore();
        }
      }
    }
  },
};

// ---------- breakdown: the one signed crossing ----------
export const breakdown = {
  a: 220.6,
  b: 229.3,
  lyrics: "lower",
  render(ctx, t) {
    const tCross = wordT("And when I really");
    const tSign = wordT("I'll do it", "sign");
    const tUnsafe = wordT("Unsafe value");
    const tGrep = wordT("Grep knows");
    const a = win(t, 220.6, 229.3, 0.6, 0.3);
    dotGrid(ctx, t, { alpha: 0.25 * a, ox: -t * 4 });
    // two domains, a single thread between them
    const pr = prog(t, tCross, 1.5, ease.inOutCubic);
    rect(ctx, 0, 0, W / 2, H, C.amber, 0.04 * a * pr);
    rect(ctx, W / 2, 0, W / 2, H, C.cyan, 0.04 * a * pr);
    text(ctx, "async", 120, 140, { size: 30, fam: F.mono, weight: 700, color: C.amber, alpha: a * pr });
    text(ctx, "sync(out_clk, async out_rst)", W - 120, 140, { size: 30, fam: F.mono, weight: 700, color: C.cyan, align: "right", alpha: a * pr });
    const s = "val in_data_domain = unsafe_value_with_domain(in_data, sync(out_clk, async out_rst));";
    const size = 30;
    const w = measure(ctx, s, size, F.mono, 400);
    const ps = prog(t, tSign - 0.2, 0.4);
    if (ps > 0) {
      const n = Math.floor(clamp((t - tSign + 0.2) / (tUnsafe - tSign + 1.2)) * s.length);
      const glowAmt = 20 + 30 * pulse(t, 3);
      ctx.save();
      ctx.shadowColor = C.red;
      ctx.shadowBlur = t > tUnsafe ? glowAmt : 0;
      drawCode(ctx, s, W / 2 - w / 2, 470, { size, alpha: a * ps, chars: n, cursor: n < s.length });
      ctx.restore();
    }
    // grep
    const pg = prog(t, tGrep - 0.1, 0.3);
    if (pg > 0) {
      panel(ctx, 260, 620, 1400, 170, { title: "~/project", alpha: a * pg });
      text(ctx, "$ grep -rn unsafe_ .", 300, 700, { size: 24, fam: F.mono, weight: 700, color: C.text, alpha: a * pg });
      const out = window.COMPILER.grep.replace(/\s+/g, " ");
      const pl = prog(t, tGrep + 0.35, 0.2);
      text(ctx, out.slice(0, 110), 300, 750, { size: 20, fam: F.mono, color: C.red, alpha: a * pl });
      text(ctx, "1 crossing. signed, greppable, reviewed.", W / 2, 880, { size: 30, fam: F.sans, weight: 700, color: C.muted, align: "center", alpha: a * prog(t, tGrep + 0.7, 0.4) });
    }
  },
};
