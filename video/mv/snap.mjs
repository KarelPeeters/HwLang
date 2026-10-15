// Render individual frames to PNG for review: node snap.mjs out_dir t1 t2 ...
import puppeteer from "puppeteer-core";
import http from "http";
import fs from "fs";
import path from "path";

const ROOT = path.resolve(import.meta.dirname, "..");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".woff2": "font/woff2", ".wav": "audio/wav", ".png": "image/png", ".jpg": "image/jpeg" };

export function serve(port = 0) {
  return new Promise((res) => {
    const srv = http.createServer((req, resp) => {
      const p = path.join(ROOT, decodeURIComponent(req.url.split("?")[0]));
      if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
        resp.writeHead(404);
        return resp.end();
      }
      resp.writeHead(200, { "Content-Type": MIME[path.extname(p)] || "application/octet-stream" });
      fs.createReadStream(p).pipe(resp);
    });
    srv.listen(port, "127.0.0.1", () => res(srv));
  });
}

export async function openPage(browser, port) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  page.on("console", (m) => (m.type() === "error" || m.type() === "warn") && console.error("[page]", m.text()));
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));
  await page.goto(`http://127.0.0.1:${port}/mv/index.html`);
  await page.waitForFunction("window.__ready === true", { timeout: 60000 });
  return page;
}

export function launch() {
  return puppeteer.launch({
    executablePath: "/usr/bin/google-chrome",
    headless: true,
    args: ["--hide-scrollbars", "--force-color-profile=srgb", "--disable-gpu-vsync", "--font-render-hinting=none"],
  });
}

if (process.argv[1] === import.meta.filename) {
  const [out, ...ts] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
  const srv = await serve();
  const browser = await launch();
  const page = await openPage(browser, srv.address().port);
  for (const t of ts) {
    await page.evaluate((t) => window.__render(t), parseFloat(t));
    await page.screenshot({ path: path.join(out, `t${parseFloat(t).toFixed(2).padStart(7, "0")}.png`) });
  }
  await browser.close();
  srv.close();
}
