// Render the music video: node render.mjs [--fps=60] [--workers=8] [--from=0] [--to=331.18] [--scale=1] [--out=../out/khdl.mp4]
// Each worker renders a contiguous frame range in headless Chrome and pipes JPEG frames into its own ffmpeg segment;
// segments are then concatenated and muxed with the audio.
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import { serve, launch, openPage } from "./snap.mjs";

const args = Object.fromEntries(process.argv.slice(2).map((a) => a.replace(/^--/, "").split("=")));
const FPS = parseInt(args.fps || "60");
const WORKERS = parseInt(args.workers || "8");
const FROM = parseFloat(args.from || "0");
const TO = parseFloat(args.to || "331.18");
const SCALE = parseFloat(args.scale || "1");
const OUT = path.resolve(args.out || "../out/khdl.mp4");
const CRF = args.crf || "16";
const segDir = path.join(path.dirname(OUT), "segments_" + path.basename(OUT, ".mp4"));
fs.mkdirSync(segDir, { recursive: true });

const f0 = Math.round(FROM * FPS);
const f1 = Math.round(TO * FPS);
const total = f1 - f0;
const per = Math.ceil(total / WORKERS);
const srv = await serve();
const port = srv.address().port;
const W = Math.round((1920 * SCALE) / 2) * 2;
const H = Math.round((1080 * SCALE) / 2) * 2;

let done = 0;
const tStart = Date.now();
const progress = setInterval(() => {
  const el = (Date.now() - tStart) / 1000;
  const rate = done / el;
  process.stdout.write(`\r${done}/${total} frames  ${rate.toFixed(1)} fps  eta ${((total - done) / Math.max(rate, 0.01) / 60).toFixed(1)} min   `);
}, 2000);

async function worker(k) {
  const a = f0 + k * per;
  const b = Math.min(f1, a + per);
  if (a >= b) return null;
  const seg = path.join(segDir, `seg_${String(k).padStart(3, "0")}.mp4`);
  const browser = await launch();
  const page = await openPage(browser, port);
  const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-",
    ...(SCALE !== 1 ? ["-vf", `scale=${W}:${H}:flags=lanczos`] : []),
    "-c:v", "libx264", "-preset", "medium", "-crf", CRF, "-pix_fmt", "yuv420p", "-r", String(FPS), seg], { stdio: ["pipe", "inherit", "inherit"] });
  const cdp = await page.createCDPSession();
  for (let f = a; f < b; f++) {
    await page.evaluate((t) => window.__render(t), f / FPS);
    const { data } = await cdp.send("Page.captureScreenshot", { format: "jpeg", quality: 94, optimizeForSpeed: true });
    const buf = Buffer.from(data, "base64");
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    done++;
  }
  ff.stdin.end();
  await new Promise((r) => ff.on("close", r));
  await browser.close();
  return seg;
}

const segs = (await Promise.all(Array.from({ length: WORKERS }, (_, k) => worker(k)))).filter(Boolean);
clearInterval(progress);
const list = path.join(segDir, "list.txt");
fs.writeFileSync(list, segs.map((s) => `file '${s}'`).join("\n"));
await new Promise((res, rej) => {
  const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list, "-ss", String(FROM), "-t", String(TO - FROM), "-i", path.resolve("../audio/khdl.wav"),
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", OUT], { stdio: "inherit" });
  ff.on("close", (c) => (c === 0 ? res() : rej(new Error("ffmpeg concat failed"))));
});
srv.close();
console.log(`\nwrote ${OUT} in ${((Date.now() - tStart) / 60000).toFixed(1)} min`);
