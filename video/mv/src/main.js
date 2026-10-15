import { C, W, H, setAudio } from "./lib.js";
import { vignette, grain } from "./draw.js";
import { initKhdl } from "./khdl.js";
import { setLyrics, drawLyrics } from "./lyrics.js";
import { SCENES } from "./scenes/index.js";

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");

const FONTS = [
  ["JetBrains Mono", "jetbrains-mono-latin-400-normal", 400],
  ["JetBrains Mono", "jetbrains-mono-latin-700-normal", 700],
  ["JetBrains Mono", "jetbrains-mono-latin-800-normal", 800],
  ["Space Grotesk", "space-grotesk-latin-400-normal", 400],
  ["Space Grotesk", "space-grotesk-latin-500-normal", 500],
  ["Space Grotesk", "space-grotesk-latin-700-normal", 700],
  ["Space Grotesk", "space-grotesk-latin-700-normal", 600],
  ["Archivo Black", "archivo-black-latin-400-normal", 400],
];

async function load() {
  await Promise.all(
    FONTS.map(async ([fam, file, weight]) => {
      const f = new FontFace(fam, `url(fonts/${file}.woff2)`, { weight: String(weight) });
      await f.load();
      document.fonts.add(f);
    }),
  );
  const [audio, lyrics, compiler] = await Promise.all(
    ["audio", "lyrics", "compiler"].map((n) => fetch(`data/${n}.json`).then((r) => r.json())),
  );
  setAudio(audio);
  setLyrics(lyrics);
  window.COMPILER = compiler;
  await initKhdl();
  for (const s of SCENES) if (s.init) await s.init();
}

export function renderFrame(t) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.shadowBlur = 0;
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  let lyricStyle = null;
  for (const s of SCENES) {
    if (t < s.a || t >= s.b) continue;
    ctx.save();
    s.render(ctx, t, (t - s.a) / (s.b - s.a));
    ctx.restore();
    if (s.lyrics !== undefined) lyricStyle = typeof s.lyrics === "function" ? s.lyrics(t) : s.lyrics;
  }
  vignette(ctx, 0.55);
  if (lyricStyle && lyricStyle !== "none") {
    ctx.save();
    drawLyrics(ctx, t, typeof lyricStyle === "string" ? { style: lyricStyle } : lyricStyle);
    ctx.restore();
  }
  grain(ctx, t, 0.035);
}

await load();
window.__render = (t) => renderFrame(t);
window.__ready = true;

const params = new URLSearchParams(location.search);
if (params.has("preview")) {
  document.body.classList.add("preview");
  const audio = new Audio("../audio/khdl.wav");
  audio.currentTime = parseFloat(params.get("t") || "0");
  const hud = document.getElementById("hud");
  document.addEventListener("keydown", (e) => {
    if (e.key === " ") audio.paused ? audio.play() : audio.pause();
    if (e.key === "ArrowLeft") audio.currentTime -= 5;
    if (e.key === "ArrowRight") audio.currentTime += 5;
  });
  canvas.addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
  const loop = () => {
    renderFrame(audio.currentTime);
    hud.textContent = `${audio.currentTime.toFixed(2)}s  [space] play/pause  [←/→] seek`;
    requestAnimationFrame(loop);
  };
  loop();
} else if (params.has("t")) {
  renderFrame(parseFloat(params.get("t")));
}
