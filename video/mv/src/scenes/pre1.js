import { C, W, H, clamp, ease, prog, win, pulse, hexA, lerp, rnd, beatInfo } from "../lib.js";
import { F, text, measure, panel, dotGrid, line, glow, rect, drawCode } from "../draw.js";
import { wordT } from "../lyrics.js";

function slot(ctx, x, y, items, t, t0, t1, { size = 120, color = C.text, w = 300, seed = 1 } = {}) {
  // reel spins between t0 and t1, landing on items[final]
  const n = items.length;
  const p = clamp((t - t0) / (t1 - t0));
  const spins = 6 + seed;
  const pos = ease.outCubic(p) * (spins * n + seed);
  const h = size * 1.2;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x - w / 2, y - h, w, h * 1.5);
  ctx.clip();
  for (let k = -1; k <= 1; k++) {
    const idx = ((Math.floor(pos) + k) % n + n) % n;
    const off = (pos - Math.floor(pos) - k) * h;
    text(ctx, items[idx], x, y + off, { size, fam: F.mono, weight: 800, color, align: "center" });
  }
  ctx.restore();
  return items[((Math.floor(pos)) % n + n) % n];
}

export const pre1 = {
  a: 63.6,
  b: 82.5,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 10 });
    const tBlock = wordT("Blocking or non-blocking");
    const tNon = wordT("Blocking or non-blocking", "non");
    const tWhich = wordT("Blocking or non-blocking", "which");
    const tWire = wordT("Is it a wire", "wire");
    const tFlop = wordT("Is it a wire", "flop");
    const tTool = wordT("Is it a wire", "tool");
    const tMonths = wordT("Three months before");
    const tRan = wordT("Three months before", "ran");
    const t41 = wordT("Forty-one");
    const tSlip = wordT("Forty-one", "slipping");

    // ---- = vs <= ----
    {
      const a = win(t, 63.6, tWire - 0.15, 0.2, 0.25);
      if (a > 0) {
        const pb = prog(t, tBlock - 0.05, 0.2, ease.outBack);
        const pn = prog(t, tNon - 0.05, 0.2, ease.outBack);
        const merge = prog(t, tWhich - 0.1, 0.35, ease.inOutCubic);
        ctx.globalAlpha = a;
        // big operators
        ctx.save();
        ctx.translate(lerp(620, 520, merge), lerp(470, 300, merge));
        ctx.scale(pb * lerp(1, 0.55, merge), pb * lerp(1, 0.55, merge));
        glow(ctx, C.cyan, 30, () => text(ctx, "=", 0, 80, { size: 300, fam: F.mono, weight: 800, color: C.cyan, align: "center" }));
        text(ctx, "blocking", 0, 190, { size: 40, fam: F.sans, weight: 700, color: C.muted, align: "center" });
        ctx.restore();
        ctx.save();
        ctx.translate(lerp(1300, 1400, merge), lerp(470, 300, merge));
        ctx.scale(pn * lerp(1, 0.55, merge), pn * lerp(1, 0.55, merge));
        glow(ctx, C.magenta, 30, () => text(ctx, "<=", 0, 80, { size: 300, fam: F.mono, weight: 800, color: C.magenta, align: "center" }));
        text(ctx, "non-blocking", 0, 190, { size: 40, fam: F.sans, weight: 700, color: C.muted, align: "center" });
        ctx.restore();
        // the code with the slot machine operator
        if (merge > 0) {
          const y = 640;
          const lines = [
            ["always @(posedge clk)", null],
            ["    q1 ", "d;", 0],
            ["    q2 ", "q1;", 1],
          ];
          ctx.globalAlpha = a * merge;
          panel(ctx, 460, y - 70, 1000, 300, { title: "pipeline.v", alpha: a * merge });
          text(ctx, "always @(posedge clk) begin", 520, y + 20, { size: 40, fam: F.mono, color: CODEC.kw });
          for (let k = 0; k < 2; k++) {
            const yy = y + 80 + k * 64;
            text(ctx, ["    q1", "    q2"][k], 520, yy, { size: 40, fam: F.mono, color: C.text });
            slot(ctx, 520 + measure(ctx, "    q1 ", 40, F.mono, 400) + 40, yy, ["=", "<="], t, tWhich + k * 0.15, tWire - 0.2, { size: 40, color: C.amber, w: 90, seed: k });
            text(ctx, ["d;", "q1;"][k], 520 + measure(ctx, "    q1 ", 40, F.mono, 400) + 100, yy, { size: 40, fam: F.mono, color: C.text });
          }
        }
      }
    }

    // ---- logic: wire? flop? latch? ----
    {
      const a = win(t, tWire - 0.2, tMonths - 0.1, 0.25, 0.25);
      if (a > 0) {
        ctx.globalAlpha = a;
        text(ctx, "logic [7:0] x;", W / 2, 360, { size: 90, fam: F.mono, weight: 700, color: C.text, align: "center" });
        const labels = ["wire?", "flop?", "latch?", "wire?", "flop?"];
        const k = Math.floor(clamp((t - tWire) / (tTool - tWire)) * 4.99);
        const lab = labels[k];
        const bx = W / 2;
        const by = 560;
        const pp = pulse(t, 10);
        panel(ctx, bx - 230, by - 90, 460, 150, { alpha: a, fill: "#151a26" });
        text(ctx, lab, bx, by + 20, { size: 84, fam: F.disp, color: [C.cyan, C.magenta, C.red, C.cyan, C.magenta][k], align: "center", alpha: a * (0.8 + 0.2 * pp) });
        // the tool's answer
        const tp = prog(t, tTool - 0.1, 0.3, ease.outBack);
        if (tp > 0) {
          ctx.save();
          ctx.translate(W / 2, 760);
          ctx.scale(tp, tp);
          panel(ctx, -360, -50, 720, 100, { alpha: a });
          text(ctx, "synthesis: ok ✓   (it guessed)", 0, 14, { size: 34, fam: F.mono, weight: 700, color: C.green, align: "center", alpha: a });
          ctx.restore();
        }
      }
    }

    // ---- CDC tool finally ran ----
    {
      const a = win(t, tMonths - 0.25, 82.5, 0.3, 0.05);
      if (a > 0) {
        const X = 160;
        const Y = 110;
        panel(ctx, X, Y, 1600, 300, { title: "signoff — cdc_check", alpha: a });
        ctx.globalAlpha = a;
        text(ctx, "$ cdc_check --full-chip --waivers waivers_v17_FINAL_final2.tcl", X + 40, Y + 90, { size: 26, fam: F.mono, color: C.text });
        const bp = clamp((t - tMonths) / (tRan - tMonths + 0.2));
        const bw = 1200;
        rect(ctx, X + 40, Y + 130, bw, 30, "#1c2230");
        rect(ctx, X + 40, Y + 130, bw * ease.inOutQuad(bp), 30, bp >= 1 ? C.green : C.cyan);
        const hours = Math.floor(lerp(0, 14, bp));
        text(ctx, `${Math.floor(bp * 100)}%   elapsed ${hours}h ${String(Math.floor(lerp(0, 59, (bp * 7) % 1))).padStart(2, "0")}m`, X + 1270, Y + 154, { size: 24, fam: F.mono, color: C.muted });
        text(ctx, "T-minus 3 months to tape-out", X + 40, Y + 220, { size: 24, fam: F.mono, color: C.amber });
        // scrolling analysis log while it runs
        const la = 1 - prog(t, t41 - 0.7, 0.3);
        if (la > 0) {
          const mods = ["u_core/u_fifo", "u_dma/u_ctrl", "u_eth/u_mac", "u_pcie/u_ltssm", "u_ddr/u_phy", "u_cpu/u_irq"];
          const sigs = ["wr_ptr_gray", "rd_en", "status_q", "irq_pending", "cfg_valid", "tx_done"];
          const clks = ["clk_core", "clk_eth", "clk_ddr", "clk_pcie"];
          const n0 = Math.floor((t - tMonths) * 22);
          for (let k = 0; k < 15; k++) {
            const n = n0 - k;
            if (n < 0) continue;
            const m = mods[Math.floor(rnd(n, 1) * 6)];
            const sg = sigs[Math.floor(rnd(n, 2) * 6)];
            const c1 = clks[Math.floor(rnd(n, 3) * 4)];
            const c2 = clks[Math.floor(rnd(n, 4) * 4)];
            const bad = rnd(n, 5) < 0.18;
            const msg = `[cdc] ${m}/${sg}[${Math.floor(rnd(n, 6) * 32)}]  ${c1} -> ${c2}  ${bad ? "VIOLATION: unsynchronized" : "ok (waived)"}`;
            text(ctx, msg, X + 40, 920 - k * 32, { size: 22, fam: F.mono, color: bad ? C.red : C.muted, alpha: a * la * (1 - k / 15) });
          }
        }
        // violations counter
        const vp = clamp((t - t41 + 0.4) / 0.6);
        if (vp > 0) {
          const n = Math.round(41 * ease.outCubic(vp));
          const sh = pulse(t, 8) * 4;
          glow(ctx, C.red, 50, () => text(ctx, String(n), 560 + sh, 690, { size: 280, fam: F.disp, color: C.red, align: "center" }));
          text(ctx, "VIOLATIONS", 560, 780, { size: 44, fam: F.disp, color: C.text, align: "center", spacing: 8 });
        }
        // gantt chart slipping
        const gp = prog(t, t41 - 0.6, 0.5);
        if (gp > 0) {
          const GX = 960;
          const GY = 480;
          const rows = ["RTL freeze", "CDC signoff", "synthesis", "place & route", "TAPE-OUT"];
          const slip = Math.max(0, t - tSlip + 0.6);
          rows.forEach((r, i) => {
            const y = GY + i * 70;
            text(ctx, r, GX, y + 28, { size: 22, fam: F.mono, color: i === 4 ? C.red : C.muted, alpha: gp });
            const base = 230 + i * 90;
            const off = Math.min(1, slip * 0.6) * (60 + i * 70) + (slip > 0 ? Math.sin(slip * 30 + i) * 2 : 0);
            rect(ctx, GX + base + off, y + 6, 160, 30, i === 4 ? C.red : i === 1 ? C.amber : "#3a4560", gp);
          });
          const todayX = GX + 330;
          line(ctx, todayX, GY - 20, todayX, GY + 340, C.red, 2, gp);
          text(ctx, "today", todayX, GY - 30, { size: 18, fam: F.mono, color: C.red, align: "center", alpha: gp });
        }
      }
    }
  },
};

const CODEC = { kw: "#c49bff" };
