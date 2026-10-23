// Figure for "Clock domains are part of the type": the signals of the example take the color of their clock domain,
// and the compiler checks the reads and assignments in the process against the domain of that process.

import {createSvg, el, progress, set, setClass, setText, visibility} from "./util.js";

const WIDTH = 420;
const HEIGHT = 292;
const DURATION = 13;

const WAVE_X0 = 64;
const WAVE_X1 = 412;
const WAVE_HALF = 7;
const PX_PER_SECOND = 44;
const PERIOD_A = 1.1;
const PERIOD_B = 1.6;

const MODULE_TOP = 72;
const MODULE_BOTTOM = 172;
const BOUNDARY_X = 150;
const WIRE_Y = 128;
const SEEN_X0 = 214;
const SEEN_X1 = 284;

// the lines of the example that the compiler checks, with what it finds
const LINE_Y = [202, 224, 246];
const LINES = [
    {indent: 0, segments: [["clocked", "tk-kw"], ["(clk_b, ", ""], ["sync", "tk-dom"], [" rst_b) {", ""]], note: "domain sync(clk_b)", ok: null},
    {indent: 1, segments: [["reg wire", "tk-kw"], [" seen = ", ""], ["false", "tk-lit"], [";", ""]], note: "seen is sync(clk_b)", ok: true},
    {indent: 1, segments: [["if", "tk-kw"], [" (request) {", ""]], note: "request is sync(clk_a)", ok: false},
];
const LINE_START = [2.8, 4.0, 5.2];
const LINE_END = [4.0, 5.2, 7.0];
const ERROR_TIME = 5.7;

// Square wave of a clock, scrolling to the left with the newest time at the right edge.
function wavePath(t, period, y) {
    const start = t - (WAVE_X1 - WAVE_X0) / PX_PER_SECOND;
    const x = time => WAVE_X0 + (time - start) * PX_PER_SECOND;
    const high = time => ((time / period) % 1 + 1) % 1 < 0.5;
    const level = time => high(time) ? y - WAVE_HALF : y + WAVE_HALF;
    let d = `M${WAVE_X0},${level(start)}`;
    for (let k = Math.ceil(start / (period / 2) + 1e-9); k * period / 2 <= t; k++) {
        const time = k * period / 2;
        d += ` H${x(time).toFixed(1)} V${level(time + 1e-6)}`;
    }
    return d + ` H${WAVE_X1}`;
}

// Time since the last rising edge of a clock.
function sinceRisingEdge(t, period) {
    return ((t / period) % 1 + 1) % 1 * period;
}

export function createDomainsViz(container) {
    const svg = createSvg(container, WIDTH, HEIGHT,
        "Two clocks, clk_a and clk_b, tick at different rates. The module is split into a clk_a and a clk_b domain. " +
        "request belongs to clk_a, the process and seen belong to clk_b. The compiler checks the process line by " +
        "line, and reports an invalid domain crossing where request is read in the clk_b process.");
    const content = el(svg, "g");

    // clocks
    const waves = [["clk_a", 22, PERIOD_A, "v-text-a", "v-sig-a"], ["clk_b", 50, PERIOD_B, "v-text-b", "v-sig-b"]].map(([name, y, period, textClass, waveClass]) => {
        el(content, "text", {x: 12, y: y + 4, class: `${textClass} v-mono`, "font-size": 13}, name);
        return {path: el(content, "path", {class: waveClass}), y, period};
    });

    // the module, split into its two domains
    const zoneA = el(content, "rect", {x: 8, y: MODULE_TOP, width: BOUNDARY_X - 8, height: MODULE_BOTTOM - MODULE_TOP, class: "v-zone-a"});
    const zoneB = el(content, "rect", {x: BOUNDARY_X, y: MODULE_TOP, width: 412 - BOUNDARY_X, height: MODULE_BOTTOM - MODULE_TOP, class: "v-zone-b"});
    el(content, "rect", {x: 8, y: MODULE_TOP, width: 404, height: MODULE_BOTTOM - MODULE_TOP, rx: 6, class: "v-frame-solid"});
    el(content, "path", {d: `M${BOUNDARY_X},${MODULE_TOP} V${MODULE_BOTTOM}`, class: "v-frame"});
    const labelA = el(content, "text", {x: 16, y: MODULE_TOP + 18, class: "v-text-a v-mono", "font-size": 12}, "sync(clk_a)");
    const labelB = el(content, "text", {x: BOUNDARY_X + 8, y: MODULE_TOP + 18, class: "v-text-b v-mono", "font-size": 12}, "sync(clk_b)");
    el(content, "text", {x: 404, y: MODULE_TOP + 18, "text-anchor": "end", class: "v-muted v-mono", "font-size": 12}, "Crossing");

    // request: from the clk_a side into the clk_b process
    el(content, "path", {d: `M8,${WIRE_Y} H${SEEN_X0}`, class: "v-sig"});
    const requestA = el(content, "path", {d: `M8,${WIRE_Y} H${BOUNDARY_X}`, class: "v-sig-a"});
    const requestB = el(content, "path", {d: `M${BOUNDARY_X},${WIRE_Y} H${SEEN_X0}`, class: "v-sig-a"});
    const requestText = el(content, "g");
    el(requestText, "text", {x: 16, y: WIRE_Y - 8, class: "v-text-a v-mono", "font-size": 12}, "request");
    el(requestText, "text", {x: 16, y: WIRE_Y + 18, class: "v-text-a v-mono", "font-size": 11}, "sync(clk_a) bool");

    // seen: the register of the clk_b process
    el(content, "path", {d: `M${SEEN_X1},${WIRE_Y} H412`, class: "v-sig"});
    const seenWire = el(content, "path", {d: `M${SEEN_X1},${WIRE_Y} H412`, class: "v-sig-b"});
    const seenBox = el(content, "rect", {x: SEEN_X0, y: WIRE_Y - 22, width: SEEN_X1 - SEEN_X0, height: 44, rx: 4, class: "v-box"});
    const seenBoxB = el(content, "rect", {x: SEEN_X0, y: WIRE_Y - 22, width: SEEN_X1 - SEEN_X0, height: 44, rx: 4, class: "v-box-b"});
    el(content, "path", {d: `M${SEEN_X0},${WIRE_Y + 8} l7,5 l-7,5`, class: "v-wire"});
    el(content, "text", {x: (SEEN_X0 + SEEN_X1) / 2, y: WIRE_Y + 5, "text-anchor": "middle", class: "v-strong v-mono", "font-size": 13}, "seen");
    const seenText = el(content, "g");
    el(seenText, "text", {x: 408, y: WIRE_Y - 8, "text-anchor": "end", class: "v-text-b v-mono", "font-size": 12}, "seen");
    el(seenText, "text", {x: 408, y: WIRE_Y + 18, "text-anchor": "end", class: "v-text-b v-mono", "font-size": 11}, "sync(clk_b) bool");

    // where the compiler finds the crossing
    const marker = el(content, "g");
    el(marker, "circle", {cx: BOUNDARY_X, cy: WIRE_Y, r: 10, class: "v-bad-mark"});
    el(marker, "text", {x: BOUNDARY_X, y: WIRE_Y + 5, "text-anchor": "middle", class: "v-red", "font-size": 14, "font-weight": 700}, "✗");

    // the checked lines of the process
    const lines = LINES.map((line, i) => {
        const highlight = el(content, "rect", {x: 8, y: LINE_Y[i] - 15, width: 404, height: 21, rx: 4, class: "v-hl", opacity: 0});
        const text = el(content, "text", {x: 16 + line.indent * 31, y: LINE_Y[i], class: "v-strong v-mono", "font-size": 13});
        for (const [str, cls] of line.segments) {
            el(text, "tspan", cls ? {class: cls} : {}, str);
        }
        const note = el(content, "text", {x: 404, y: LINE_Y[i], "text-anchor": "end", class: "v-mono", "font-size": 12});
        return {highlight, text, note};
    });
    const error = el(content, "text", {x: 12, y: HEIGHT - 10, class: "v-red v-mono", "font-size": 12, "font-weight": 650},
        "error: invalid domain crossing: different clock");

    function render(t) {
        for (const wave of waves) {
            set(wave.path, "d", wavePath(t, wave.period, wave.y));
        }

        // every signal gets the color of its domain, each domain flashes briefly on its clock edges
        const domainA = progress(t, 0.4, 1.2);
        const domainB = progress(t, 1.4, 2.2);
        const flash = period => 0.06 * Math.exp(-sinceRisingEdge(t, period) * 5);
        set(zoneA, "opacity", domainA * (0.07 + flash(PERIOD_A)));
        set(zoneB, "opacity", domainB * (0.07 + flash(PERIOD_B)));
        set(labelA, "opacity", domainA);
        set(requestA, "opacity", domainA);
        set(requestText, "opacity", domainA);
        set(labelB, "opacity", domainB);
        set(seenWire, "opacity", domainB);
        set(seenBoxB, "opacity", domainB);
        set(seenBox, "opacity", 1 - domainB);
        set(seenText, "opacity", domainB);

        // the compiler checks the process line by line
        lines.forEach((line, i) => {
            set(line.highlight, "opacity", visibility(t, LINE_START[i], LINE_END[i], 0.15));
            const shown = t >= LINE_START[i] + 0.4;
            const l = LINES[i];
            setText(line.note, shown ? (l.ok === null ? l.note : `${l.note} ${l.ok ? "✓" : "✗"}`) : "");
            setClass(line.note, `v-mono ${l.ok === false ? "v-red" : l.ok ? "v-green" : "v-text-b"}`);
        });

        // the crossing
        const crossing = t >= ERROR_TIME;
        setClass(requestB, crossing ? "v-sig-bad" : "v-sig-a");
        set(requestB, "opacity", crossing ? 1 : domainA);
        set(marker, "opacity", progress(t, ERROR_TIME, ERROR_TIME + 0.3));
        set(error, "opacity", progress(t, ERROR_TIME + 0.6, ERROR_TIME + 1.0));

        set(content, "opacity", visibility(t, 0, DURATION, 0.5));
    }

    return {duration: DURATION, keyframes: [2.4, 3.6, 4.8, 6.8, 10], render};
}
