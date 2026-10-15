// Capture the real KHDL playground (local build of web_demo) while code is typed into it.
import puppeteer from "puppeteer-core";
import fs from "fs";

const OUT = "assets/playground";
const browser = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome", headless: true, args: ["--hide-scrollbars"] });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 1 });
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
await page.goto("http://127.0.0.1:8765/");
await page.waitForSelector("#div-editor-input .cm-content", { timeout: 30000 });
await new Promise((r) => setTimeout(r, 1500));
await page.screenshot({ path: `${OUT}/initial.png` });

// clear and type our own module, frame by frame
await page.click("#div-editor-input .cm-content");
await page.keyboard.down("Control");
await page.keyboard.press("a");
await page.keyboard.up("Control");
await page.keyboard.press("Backspace");
const src = `module top ports(
    clk: in clock,
    rst: in async bool,
    x_clk: in clock,
    x_data: in sync(x_clk) uint(0..8),
    sync(clk, async rst) {
        y: out uint(0..16),
    }
) {
    clocked(clk, async rst) {
        reg wire y = 0;
        y = x_data + 1;
    }
}`;
let n = 0;
const frames = [];
for (const ch of src) {
  if (ch === "\n") {
    await page.keyboard.press("Enter");
    // the editor copies the previous line's indentation; remove it so we control indentation
    const prevLine = src.slice(0, n).split("\n").pop();
    if (/^ +/.test(prevLine)) {
      await page.keyboard.down("Shift");
      await page.keyboard.press("Home");
      await page.keyboard.up("Shift");
      await page.keyboard.press("Backspace");
    }
  } else await page.keyboard.type(ch);
  // codemirror auto-indents; closing braces are typed as-is
  if (n % 3 === 0 || ch === "\n") {
    const f = `${OUT}/type_${String(frames.length).padStart(4, "0")}.png`;
    await page.screenshot({ path: f });
    frames.push(f);
  }
  n++;
}
await new Promise((r) => setTimeout(r, 500));
for (let k = 0; k < 6; k++) {
  const f = `${OUT}/type_${String(frames.length).padStart(4, "0")}.png`;
  await page.screenshot({ path: f });
  frames.push(f);
}
// the fix: put x_data in the right domain, then capture the Verilog view
await page.click("#div-editor-input .cm-content");
await page.keyboard.down("Control");
await page.keyboard.press("a");
await page.keyboard.up("Control");
await page.keyboard.press("Backspace");
const fixed = src.replace("    x_clk: in clock,\n", "").replace("sync(x_clk)", "sync(clk, async rst)");
for (let i = 0; i < fixed.length; i++) {
  const ch = fixed[i];
  if (ch === "\n") {
    await page.keyboard.press("Enter");
    const prevLine = fixed.slice(0, i).split("\n").pop();
    if (/^ +/.test(prevLine)) {
      await page.keyboard.down("Shift");
      await page.keyboard.press("Home");
      await page.keyboard.up("Shift");
      await page.keyboard.press("Backspace");
    }
  } else await page.keyboard.type(ch);
}
await new Promise((r) => setTimeout(r, 600));
await page.screenshot({ path: `${OUT}/fixed.png` });
fs.writeFileSync(`${OUT}/frames.json`, JSON.stringify({ count: frames.length }));
await browser.close();
console.log("frames", frames.length);
