import { C, W, H, clamp, ease, prog, win, pulse, hexA, lerp, rnd, beatInfo } from "../lib.js";
import { F, text, measure, panel, dotGrid, line, glow, rect, busLane, wave, drawCode, rrect } from "../draw.js";
import { wordT } from "../lyrics.js";

const hex2 = (v) => v.toString(16).toUpperCase().padStart(2, "0");

// ---------- 3 a.m. waveform viewer ----------
const A_VALS = [0x2a, 0x64, 0xc8, 0x9f, 0x31, 0x77];
const B_VALS = [0x10, 0x22, 0x64, 0x41, 0x0c, 0x18];

function waveViewer(ctx, t) {
  const tStart = 26.0;
  const tCarry = wordT("A plus B", "carry");
  const tDrop = wordT("Verilog just dropped", "dropped");
  const tTop = wordT("Verilog just dropped", "top");
  const tCtx = wordT("Context-determined");
  const tFeat = wordT("Context-determined", "feature");
  const tGag = wordT("Context-determined", "gag");
  const a = win(t, tStart, 41.2, 0.7, 0.35);
  if (a <= 0) return;
  const zoomIn = 1 - ease.outCubic(clamp((t - tStart) / 1.2));
  ctx.save();
  ctx.translate(W / 2, H / 2);
  ctx.scale(1 + zoomIn * 0.25, 1 + zoomIn * 0.25);
  ctx.translate(-W / 2, -H / 2);

  const X = 110;
  const Y = 90;
  const PW = 1700;
  const PH = 640;
  panel(ctx, X, Y, PW, PH, { title: "wave viewer — dump.vcd", alpha: a });
  ctx.globalAlpha = a;
  // 3 a.m. clock in the title bar
  const blink = Math.floor(t * 2) % 2 ? ":" : " ";
  text(ctx, `03${blink}00`, X + PW - 30, Y + 25, { size: 18, fam: F.mono, weight: 700, color: C.amber, align: "right", alpha: a });

  const names = ["clk", "a[7:0]", "b[7:0]", "sum[7:0]"];
  const lx = X + 260;
  const cw = 230;
  const laneY = (i) => Y + 100 + i * 115;
  line(ctx, lx - 20, Y + 50, lx - 20, Y + PH - 20, C.rule, 1, a);
  names.forEach((n, i) => {
    const r = prog(t, tStart + 0.4 + i * 0.35, 0.5);
    text(ctx, n, X + 40, laneY(i) + 34, { size: 24, fam: F.mono, weight: 700, color: i === 3 ? C.text : C.muted, alpha: a * r });
    if (i === 0) wave(ctx, lx, laneY(0) + 48, 6 * cw, 44, { period: cw, phase: 0, color: C.cyan, reveal: r, alpha: a, glowAmt: 6 });
    else {
      const vals = i === 1 ? A_VALS : i === 2 ? B_VALS : A_VALS.map((v, k) => (v + B_VALS[k]) & 0xff);
      const cells = vals.map((v, k) => ({ x0: k * cw, x1: (k + 1) * cw, label: `0x${hex2(v)}`, color: i === 3 && k === 2 ? C.red : undefined }));
      busLane(ctx, lx, laneY(i), 6 * cw, 56, cells, { color: i === 3 ? C.green : "#7fa0c8", reveal: r, alpha: a, size: 22 });
    }
  });

  // cursor on cycle 2
  const cx = lx + 2.5 * cw;
  const cr = prog(t, tStart + 2.2, 0.6);
  line(ctx, cx, Y + 60, cx, lerp(Y + 60, Y + PH - 30, cr), C.amber, 2, a);
  if (cr > 0.5) {
    rect(ctx, cx - 70, Y + 46, 140, 28, C.amber, a);
    text(ctx, "t = 3:00 AM", cx, Y + 66, { size: 16, fam: F.mono, weight: 700, color: "#000", align: "center", alpha: a });
  }

  // expected vs actual
  const ex = prog(t, wordT("Three a.m.", "sum"), 0.4);
  if (ex > 0) {
    const yy = laneY(3) + 100;
    text(ctx, "expected 0xC8 + 0x64 = 0x12C", cx, yy, { size: 22, fam: F.mono, weight: 700, color: C.muted, align: "center", alpha: a * ex });
  }

  // the carry bit escapes: pops out of the sum cell and falls out of frame
  const sumY = laneY(3);
  const pc = t - tCarry;
  if (pc > -0.05) {
    const g = 1600;
    let bx = cx + 40 + Math.max(0, pc) * 120;
    let by;
    const hop = 0.35;
    if (pc < hop) by = sumY + 28 - Math.sin((Math.max(0, pc) / hop) * Math.PI) * 120;
    else {
      const ft = pc - hop;
      by = sumY + 28 + 0.5 * g * ft * ft * (t < tDrop ? 0.0 : 1);
      if (t < tDrop) by = sumY + 28 + Math.sin((pc - hop) * 4) * 10 - 30; // dangling, held by a thread
      else {
        const dt = t - tDrop;
        by = sumY - 2 + 0.5 * g * dt * dt;
        bx += dt * 80;
      }
    }
    const rot = t > tDrop ? (t - tDrop) * 4 : Math.sin(pc * 5) * 0.2;
    ctx.save();
    ctx.translate(bx, by);
    ctx.rotate(rot);
    glow(ctx, C.red, 30, () => {
      rect(ctx, -34, -34, 68, 68, "#2a0f16", a);
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 3;
      ctx.globalAlpha = a;
      ctx.strokeRect(-34, -34, 68, 68);
      text(ctx, "1", 0, 18, { size: 50, fam: F.mono, weight: 800, color: C.red, align: "center", alpha: a });
    });
    text(ctx, "bit 8", 0, -46, { size: 18, fam: F.mono, weight: 700, color: C.red, align: "center", alpha: a });
    ctx.restore();
  }
  // "top bit" label
  if (t > tTop) {
    const p = prog(t, tTop, 0.3);
    text(ctx, "0x12C → 0x2C", cx, sumY + 150, { size: 44, fam: F.mono, weight: 800, color: C.red, alpha: a * p, align: "center" });
    text(ctx, "0 errors, 0 warnings", cx, sumY + 190, { size: 22, fam: F.mono, weight: 500, color: C.muted, alpha: a * p, align: "center" });
  }
  ctx.restore();

  // sticky note: "context-determined width" — a feature!
  const sn = prog(t, tCtx - 0.1, 0.35, ease.outBack);
  if (sn > 0) {
    ctx.save();
    ctx.translate(1490, 610);
    ctx.rotate(-0.06);
    ctx.scale(sn, sn);
    ctx.globalAlpha = a;
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 30;
    ctx.shadowOffsetY = 12;
    ctx.fillStyle = "#ffd65a";
    ctx.fillRect(-230, -130, 460, 260);
    ctx.shadowColor = "transparent";
    text(ctx, "IEEE 1364 §5.4.1", -200, -82, { size: 20, fam: F.mono, weight: 700, color: "#6b5200" });
    text(ctx, "context-determined", -200, -26, { size: 40, fam: F.disp, color: "#2a2000" });
    text(ctx, "width", -200, 22, { size: 40, fam: F.disp, color: "#2a2000" });
    text(ctx, "operands are sized by the LHS", -200, 70, { size: 20, fam: F.sans, weight: 500, color: "#4a3a00" });
    ctx.restore();
    // FEATURE stamp
    const st = prog(t, tFeat - 0.05, 0.18, ease.outExpo);
    if (st > 0) {
      ctx.save();
      ctx.translate(1560, 770);
      ctx.rotate(-0.12);
      const sc = lerp(2.4, 1, st);
      ctx.scale(sc, sc);
      ctx.globalAlpha = a * st;
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 6;
      ctx.strokeRect(-150, -46, 300, 92);
      text(ctx, "FEATURE ✓", 0, 18, { size: 50, fam: F.disp, color: C.green, align: "center" });
      ctx.restore();
    }
    const gg = prog(t, tGag, 0.2, ease.outBack);
    if (gg > 0) {
      ctx.save();
      ctx.translate(1300, 470);
      ctx.rotate(-0.15);
      ctx.scale(gg, gg);
      text(ctx, "(it's a gag)", 0, 0, { size: 30, fam: F.sans, weight: 700, color: C.muted, align: "center", alpha: a });
      ctx.restore();
    }
  }
}

// ---------- sensitivity list ----------
function sensitivity(ctx, t) {
  const t0 = wordT("I left one signal");
  const tSim = wordT("The sim says", "sim");
  const tBoard = wordT("The sim says", "board");
  const tHow = wordT("The sim says", "nobody");
  const a = win(t, t0 - 0.3, wordT("And a missing else") - 0.1, 0.35, 0.3);
  if (a <= 0) return;
  const src = "always @(a or b)\n    y = (a & b) | c;";
  const mv = prog(t, tSim - 0.4, 0.5, ease.inOutCubic);
  const zs = lerp(1.35, 1, mv);
  ctx.save();
  ctx.translate(lerp(W / 2, 160 + 490, mv), lerp(400, 270, mv));
  ctx.scale(zs, zs);
  ctx.translate(-(160 + 490), -270);
  const X = 160;
  const Y = 120;
  panel(ctx, X, Y, 980, 300, { title: "glue_logic.v", alpha: a });
  drawCode(ctx, src, X + 50, Y + 120, { size: 44, lang: "verilog", alpha: a, chars: Math.floor((t - t0 + 0.3) * 60) });
  const cw = measure(ctx, "M", 44, F.mono, 400);
  // highlight sensitivity list and the missing c
  const hp = prog(t, wordT("I left one signal", "sensitivity"), 0.3);
  if (hp > 0) {
    ctx.save();
    ctx.globalAlpha = a * hp;
    ctx.strokeStyle = C.amber;
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(X + 50 + 7 * cw - 8, Y + 80, 9 * cw + 16, 56);
    ctx.setLineDash([]);
    const cx = X + 50 + 19 * cw;
    ctx.strokeStyle = C.red;
    ctx.beginPath();
    ctx.arc(cx + cw / 2, Y + 120 + 64 - 16, 34, 0, Math.PI * 2);
    ctx.stroke();
    text(ctx, "c is not in the list", X + 50 + 13 * cw, Y + 250, { size: 26, fam: F.mono, weight: 700, color: C.red, alpha: 1 });
    ctx.restore();
  }
  ctx.restore();
  // sim vs board cards
  const card = (x, label, val, col, tt, chip) => {
    const p = prog(t, tt - 0.05, 0.3, ease.outBack);
    if (p <= 0) return;
    const shake = t > tHow ? Math.sin(t * 50) * 6 * Math.exp(-(t - tHow) * 2) : 0;
    ctx.save();
    ctx.translate(x + shake, 640);
    ctx.scale(p, p);
    ctx.globalAlpha = a;
    panel(ctx, -300, -170, 600, 340, { alpha: 1 });
    text(ctx, label, 0, -100, { size: 30, fam: F.sans, weight: 700, color: C.muted, align: "center", spacing: 6 });
    glow(ctx, col, 40, () => text(ctx, `y = ${val}`, 0, 60, { size: 130, fam: F.mono, weight: 800, color: col, align: "center" }));
    if (chip) {
      // tiny FPGA glyph
      ctx.strokeStyle = C.muted;
      ctx.lineWidth = 2;
      ctx.strokeRect(190, -140, 70, 70);
      for (let k = 0; k < 5; k++) {
        line(ctx, 198 + k * 14, -150, 198 + k * 14, -140, C.muted, 2);
        line(ctx, 198 + k * 14, -70, 198 + k * 14, -60, C.muted, 2);
      }
    }
    ctx.restore();
  };
  card(560, "SIMULATION", 0, C.cyan, tSim, false);
  card(1360, "ON THE BOARD", 1, C.magenta, tBoard, true);
  if (t > tHow) {
    const p = prog(t, tHow, 0.3, ease.outBack);
    text(ctx, "≠", 960, 680, { size: 160, fam: F.disp, color: C.red, align: "center", alpha: a * p });
  }
}

// ---------- the latch that moved in ----------
function latch(ctx, t) {
  const t0 = wordT("And a missing else");
  const tElse = wordT("And a missing else", "else");
  const tGrew = wordT("Grew a latch");
  const tLiving = wordT("Grew a latch", "living");
  const a = win(t, t0 - 0.2, wordT("So I tried VHDL") - 0.05, 0.3, 0.3);
  if (a <= 0) return;
  const src = "always @(*) begin\n    case (state)\n        IDLE: if (start) next = RUN;\n        RUN:  next = DONE;\n    endcase\nend";
  const X = 110;
  const Y = 130;
  panel(ctx, X, Y, 860, 420, { title: "fsm.v", alpha: a });
  drawCode(ctx, src, X + 40, Y + 100, { size: 30, lang: "verilog", alpha: a });
  const cw = measure(ctx, "M", 30, F.mono, 400);
  // ghost "else" that isn't there
  const ep = prog(t, tElse - 0.1, 0.3);
  if (ep > 0) {
    const gx = X + 40 + 37 * cw;
    const gy = Y + 100 + 2 * 30 * 1.45;
    ctx.save();
    ctx.globalAlpha = a * ep * (0.5 + 0.5 * Math.sin(t * 8) ** 2);
    ctx.strokeStyle = C.red;
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 2;
    ctx.strokeRect(gx - 8, gy - 30, 6 * cw + 16, 42);
    ctx.setLineDash([]);
    text(ctx, "else ?", gx, gy, { size: 30, fam: F.mono, weight: 700, color: C.red });
    ctx.restore();
  }

  // netlist on the right with a latch that grows like an organism
  const NX = 1060;
  const NY = 130;
  const NW = 760;
  const NH = 560;
  panel(ctx, NX, NY, NW, NH, { title: "synthesized netlist", alpha: a });
  ctx.save();
  ctx.beginPath();
  ctx.rect(NX, NY + 37, NW, NH - 37);
  ctx.clip();
  // gates
  const gates = [
    [NX + 120, NY + 170],
    [NX + 120, NY + 330],
    [NX + 280, NY + 250],
    [NX + 600, NY + 470],
  ];
  for (const [gx, gy] of gates) {
    ctx.globalAlpha = a * 0.7;
    ctx.strokeStyle = C.muted;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(gx - 30, gy - 26);
    ctx.lineTo(gx, gy - 26);
    ctx.arc(gx, gy, 26, -Math.PI / 2, Math.PI / 2);
    ctx.lineTo(gx - 30, gy + 26);
    ctx.closePath();
    ctx.stroke();
  }
  line(ctx, NX + 146, NY + 170, NX + 254, NY + 240, C.muted, 2, a * 0.7);
  line(ctx, NX + 146, NY + 330, NX + 254, NY + 260, C.muted, 2, a * 0.7);
  // latch
  const g = prog(t, tGrew - 0.05, 0.8, ease.outElastic);
  const lx = NX + 470;
  const ly = NY + 250;
  if (g > 0) {
    // vines
    const vp = clamp((t - tGrew) / 3.0);
    ctx.globalAlpha = a;
    for (let k = 0; k < 9; k++) {
      const ang = (k / 9) * Math.PI * 2 + 0.3;
      const len = 60 + rnd(k, 3) * 260;
      ctx.strokeStyle = hexA(C.red, 0.7);
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      const steps = 40;
      for (let s = 1; s <= Math.floor(steps * vp); s++) {
        const f = s / steps;
        const r = 70 + f * len;
        const w = Math.sin(f * 9 + k + t * 1.5) * 0.35;
        ctx.lineTo(lx + Math.cos(ang + w) * r, ly + Math.sin(ang + w) * r);
      }
      ctx.stroke();
    }
    ctx.save();
    ctx.translate(lx, ly);
    const breathe = 1 + 0.04 * Math.sin(t * 6);
    ctx.scale(g * breathe, g * breathe);
    glow(ctx, C.red, 50, () => {
      rect(ctx, -90, -80, 180, 160, "#2a0f16");
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 4;
      ctx.strokeRect(-90, -80, 180, 160);
    });
    text(ctx, "LATCH", 0, 14, { size: 40, fam: F.disp, color: C.red, align: "center" });
    text(ctx, "D", -76, -50, { size: 18, fam: F.mono, color: C.muted });
    text(ctx, "EN", -76, 64, { size: 18, fam: F.mono, color: C.muted });
    text(ctx, "Q", 62, -50, { size: 18, fam: F.mono, color: C.muted });
    // it moved in: a little roof and a chimney
    const rp = prog(t, tLiving - 0.1, 0.4, ease.outBack);
    if (rp > 0) {
      ctx.globalAlpha = rp;
      ctx.fillStyle = C.red;
      ctx.beginPath();
      ctx.moveTo(-110, -80);
      ctx.lineTo(0, -80 - 80 * rp);
      ctx.lineTo(110, -80);
      ctx.closePath();
      ctx.fill();
      ctx.fillRect(48, -140, 22, 40);
      for (let s = 0; s < 4; s++) {
        const ph = (t * 0.8 + s * 0.25) % 1;
        ctx.globalAlpha = rp * (1 - ph) * 0.5;
        ctx.fillStyle = C.muted;
        ctx.beginPath();
        ctx.arc(59 + Math.sin(ph * 6) * 10, -150 - ph * 70, 6 + ph * 10, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = rp;
      text(ctx, "est. last May", 0, 120, { size: 20, fam: F.mono, color: C.muted, align: "center" });
    }
    ctx.restore();
  }
  ctx.restore();
  // synthesis warning nobody reads
  const wp = prog(t, tGrew + 0.6, 0.3);
  text(ctx, "Warning (10240): inferring latch(es) for \"next\"   [1 of 4,812 warnings]", NX + 20, NY + NH + 40, { size: 18, fam: F.mono, color: C.amber, alpha: a * wp });
}

// ---------- VHDL: say it three times ----------
const PORTS = ["clk      : in  std_logic;", "rst      : in  std_logic;", "data_in  : in  std_logic_vector(7 downto 0);", "valid_in : in  std_logic;", "ready_in : out std_logic;", "data_out : out std_logic_vector(7 downto 0);"];
const VHDL = [
  "entity adder is\n  port (\n" + PORTS.map((p) => "    " + p).join("\n") + "\n  );\nend entity;",
  "component adder is\n  port (\n" + PORTS.map((p) => "    " + p).join("\n") + "\n  );\nend component;",
  "u_adder : adder\n  port map (\n" + ["clk", "rst", "data_in", "valid_in", "ready_in", "data_out"].map((p) => `    ${p.padEnd(8)} => ${p},`).join("\n") + "\n  );",
];

function vhdl(ctx, t) {
  const t0 = wordT("So I tried VHDL");
  const tThree = wordT("So I tried VHDL", "three");
  const tE = wordT("Entity", "entity");
  const tC = wordT("Entity", "component");
  const tP = wordT("Entity", "port");
  const tLong = wordT("Entity", "longer");
  const end = wordT("Blocking or non-blocking") - 0.05;
  const a = win(t, t0 - 0.2, end, 0.3, 0.25);
  if (a <= 0) return;
  // "x3" counter early
  const c3 = prog(t, tThree - 0.05, 0.25, ease.outBack);
  if (t < tE) {
    const vp = prog(t, t0, 0.3, ease.outExpo);
    ctx.save();
    ctx.globalAlpha = a * vp * (1 - c3 * 0.85);
    ctx.lineWidth = 3;
    ctx.strokeStyle = C.muted;
    ctx.font = `400 300px "${F.disp}"`;
    ctx.textAlign = "center";
    ctx.strokeText("VHDL", W / 2, 560);
    ctx.restore();
    ctx.save();
    ctx.translate(W / 2, 420);
    ctx.scale(c3, c3);
    text(ctx, "× 3", 0, 60, { size: 240, fam: F.disp, color: C.text, align: "center", alpha: a });
    ctx.restore();
    text(ctx, "library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;", W / 2, 160, { size: 26, fam: F.mono, color: C.muted, align: "center", alpha: a * prog(t, t0, 0.4) });
  }
  const times = [tE, tC, tP];
  const scroll = Math.max(0, t - tLong) * 900;
  for (let k = 0; k < 9; k++) {
    const i = k % 3;
    const tt = k < 3 ? times[i] : tLong + (k - 3) * 0.18;
    const p = prog(t, tt - 0.04, 0.22, ease.outExpo);
    if (p <= 0) continue;
    const col = k % 3;
    const row = Math.floor(k / 3);
    const x = 90 + col * 590;
    const y = 120 + row * 470 - scroll + (1 - p) * -200;
    panel(ctx, x, y, 560, 440, { title: ["adder.vhd", "top.vhd", "top.vhd"][i], alpha: a * p });
    drawCode(ctx, VHDL[i], x + 24, y + 80, { size: 17.5, lang: "verilog", alpha: a * p });
  }
  // line counter
  if (t > tE) {
    const n = Math.round(lerp(0, 27, prog(t, tE, tP - tE + 0.3, ease.linear)) + Math.max(0, t - tLong) * 140);
    rect(ctx, W - 640, 52, 570, 56, C.bg, a * 0.9);
    text(ctx, `${n} lines to connect one adder`, W - 90, 90, { size: 26, fam: F.mono, weight: 700, color: C.amber, align: "right", alpha: a });
  }
}

export const verse1 = {
  a: 25.8,
  b: 63.9,
  lyrics: "lower",
  render(ctx, t) {
    dotGrid(ctx, t, { alpha: 0.5, ox: -t * 8 });
    waveViewer(ctx, t);
    sensitivity(ctx, t);
    latch(ctx, t);
    vhdl(ctx, t);
  },
};
