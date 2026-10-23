// Figure for "Integers know their range": `count + 1` doesn't fit `uint(8)`, and the `if` splits the range of `count`
// so that both branches fit.

import {createSvg, el, lerp, progress, set, setClass, setText, visibility} from "./util.js";

const WIDTH = 420;
const HEIGHT = 304;
const DURATION = 14;

const CELL = 30;
const X0 = 30;
const ROWS = [92, 150, 208];
const BAR_HEIGHT = 24;
const FRAME_TOP = 68;
const AXIS_Y = 246;

// Value to x position of the left edge of its cell. Values 5 to 250 are compressed into a one cell wide gap.
function x(v) {
    if (v <= 5) return X0 + v * CELL;
    if (v <= 251) return X0 + 5 * CELL + (v - 5) / 246 * CELL;
    return X0 + (6 + v - 251) * CELL;
}

const FRAME_LEFT = x(0);
const FRAME_RIGHT = x(256);
const GAP_LEFT = x(5);
const GAP_RIGHT = x(251);

// Code shown above the bars, as [text, token class] segments.
const CAPTIONS = [
    [[["count", ""], [": ", ""], ["out", "tk-kw"], [" ", ""], ["uint", "tk-type"], ["(", ""], ["8", "tk-num"], [")", ""]]],
    [[["count = count + ", ""], ["1", "tk-num"], [";", ""]]],
    [
        [["if", "tk-kw"], [" (count == ", ""], ["255", "tk-num"], [") { count = ", ""], ["0", "tk-num"], ["; }", ""]],
        [["else", "tk-kw"], [" { count = count + ", ""], ["1", "tk-num"], ["; }", ""]],
    ],
];

function createCaption(parent, lines) {
    const g = el(parent, "g");
    const rows = lines.map((segments, i) => {
        const y = 24 + i * 24;
        const highlight = el(g, "rect", {x: X0 - 8, y: y - 16, width: 300, height: 22, rx: 4, class: "v-hl", opacity: 0});
        const text = el(g, "text", {x: X0, y, class: "v-strong v-mono", "font-size": 15});
        for (const [str, cls] of segments) {
            el(text, "tspan", cls ? {class: cls} : {}, str);
        }
        return highlight;
    });
    return {g, highlights: rows};
}

function createBar(parent) {
    const g = el(parent, "g");
    return {
        g,
        left: el(g, "rect", {height: BAR_HEIGHT, rx: 3}),
        right: el(g, "rect", {height: BAR_HEIGHT, rx: 3}),
        over: el(g, "rect", {height: BAR_HEIGHT, rx: 3, class: "v-bar bad"}),
        dots: el(g, "text", {"text-anchor": "middle", class: "v-muted", "font-size": 16}, "⋯"),
        label: el(g, "text", {x: X0 + 4, class: "v-text v-mono", "font-size": 14}),
    };
}

function setRect(rect, x1, x2, y) {
    set(rect, "x", x1);
    set(rect, "y", y);
    set(rect, "width", Math.max(0, x2 - x1));
    set(rect, "opacity", x2 - x1 > 0.5 ? 1 : 0);
}

// Draw a bar covering [x1, x2]. The part right of the type's frame is drawn as an overflow. Bars that cover values on
// both sides of the gap are split around it, unless `split` is false.
function drawBar(bar, {x1, x2, y, opacity, state = "", label, split = true}) {
    set(bar.g, "opacity", opacity);
    const cls = `v-bar ${state}`.trim();
    setClass(bar.left, cls);
    setClass(bar.right, cls);

    const inside = Math.min(x2, FRAME_RIGHT);
    if (split) {
        setRect(bar.left, x1, Math.min(inside, GAP_LEFT), y);
        setRect(bar.right, Math.max(x1, GAP_RIGHT), inside, y);
    } else {
        setRect(bar.left, x1, inside, y);
        setRect(bar.right, 0, 0, y);
    }
    setRect(bar.over, Math.max(x1, FRAME_RIGHT), x2, y);

    set(bar.dots, "x", (GAP_LEFT + GAP_RIGHT) / 2);
    set(bar.dots, "y", y + BAR_HEIGHT / 2 + 5);
    set(bar.dots, "opacity", split && x1 < GAP_LEFT && x2 > GAP_RIGHT ? 1 : 0);

    set(bar.label, "y", y - 7);
    setText(bar.label, label);
}

export function createRangesViz(container) {
    const svg = createSvg(container, WIDTH, HEIGHT,
        "Number line from 0 to 256. The range of count, 0 to 255, fits uint(8). count + 1 ranges from 1 to 256, " +
        "and 256 sticks out of uint(8). With the if, count is 255 in one branch and becomes 0, and 0 to 254 in the " +
        "other, where count + 1 is 1 to 255. Both fit.");

    const captions = CAPTIONS.map(lines => createCaption(svg, lines));

    // type frame and axis
    el(svg, "rect", {x: FRAME_LEFT, y: FRAME_TOP, width: FRAME_RIGHT - FRAME_LEFT, height: AXIS_Y - FRAME_TOP, class: "v-frame"});
    el(svg, "text", {x: FRAME_RIGHT - 4, y: FRAME_TOP + 15, "text-anchor": "end", class: "v-muted v-mono", "font-size": 13}, "uint(8)");
    el(svg, "path", {d: `M${x(0)},${AXIS_Y} H${GAP_LEFT - 4} M${GAP_RIGHT + 4},${AXIS_Y} H${x(257)}`, class: "v-line"});
    for (const v of [0, 1, 2, 3, 4, 5, 251, 252, 253, 254, 255, 256, 257]) {
        el(svg, "path", {d: `M${x(v)},${AXIS_Y} v5`, class: "v-line"});
    }
    let label256 = null;
    for (const v of [0, 1, 2, 3, 4, 251, 252, 253, 254, 255, 256]) {
        const label = el(svg, "text", {x: x(v) + CELL / 2, y: AXIS_Y + 20, "text-anchor": "middle", "font-size": 13, class: "v-muted v-mono"}, String(v));
        if (v === 256) label256 = label;
    }
    el(svg, "text", {x: (GAP_LEFT + GAP_RIGHT) / 2, y: AXIS_Y + 20, "text-anchor": "middle", class: "v-muted", "font-size": 16}, "⋯");

    const bars = Array.from({length: 4}, () => createBar(svg));
    const status = el(svg, "text", {x: X0, y: HEIGHT - 8, "font-size": 15});

    function render(t) {
        // captions: declaration, then the plain increment, then the if/else
        const captionOpacity = [visibility(t, -1, 1.7), visibility(t, 1.7, 5.7), visibility(t, 5.7, DURATION + 1)];
        captions.forEach((caption, i) => set(caption.g, "opacity", captionOpacity[i]));
        set(captions[1].highlights[0], "opacity", visibility(t, 1.9, 5.4));
        set(captions[2].highlights[1], "opacity", visibility(t, 6.9, 8.0, 0.25));
        set(captions[2].highlights[0], "opacity", visibility(t, 8.1, 9.2, 0.25));

        // count itself
        drawBar(bars[0], {x1: x(0), x2: x(256), y: ROWS[0], opacity: visibility(t, 0.2, 13.4), label: "count: int(0..256)"});

        // count + 1, without the if
        const shift = progress(t, 2.7, 3.4);
        drawBar(bars[1], {
            x1: x(shift), x2: x(256 + shift),
            y: lerp(ROWS[0], ROWS[1], progress(t, 1.8, 2.5)),
            opacity: visibility(t, 1.8, 5.6),
            label: shift < 0.5 ? "count + 1" : "count + 1: int(1..257)",
        });

        // else branch: count is at most 254, so count + 1 fits
        const elseShift = progress(t, 7.0, 7.8);
        drawBar(bars[2], {
            x1: x(elseShift), x2: x(255 + elseShift),
            y: lerp(ROWS[0], ROWS[1], progress(t, 6.0, 6.7)),
            opacity: visibility(t, 6.0, 13.4),
            state: elseShift >= 1 ? "good" : "",
            label: elseShift < 0.5 ? "else: count is int(0..255)" : "count + 1: int(1..256)",
        });

        // if branch: count is 255, and becomes 0
        const wrap = progress(t, 8.2, 9.0);
        const wrapX = lerp(x(255), x(0), wrap);
        drawBar(bars[3], {
            x1: wrapX, x2: wrapX + CELL,
            y: lerp(ROWS[0], ROWS[2], progress(t, 6.0, 6.7)),
            opacity: visibility(t, 6.0, 13.4),
            state: wrap >= 1 ? "good" : "",
            label: wrap < 0.5 ? "if: count == 255" : "count = 0",
            split: false,
        });

        const overflowing = t >= 3.4 && t < 5.6;
        setClass(label256, overflowing ? "v-red v-mono" : "v-muted v-mono");
        if (overflowing) {
            setText(status, "✗ 256 doesn't fit uint(8)");
            setClass(status, "v-red");
            set(status, "opacity", visibility(t, 3.4, 5.6, 0.3));
        } else if (t >= 9.2) {
            setText(status, "✓ both branches fit uint(8)");
            setClass(status, "v-green");
            set(status, "opacity", visibility(t, 9.2, 13.4, 0.3));
        } else {
            setText(status, "count can be any value from 0 to 255");
            setClass(status, "v-muted");
            set(status, "opacity", visibility(t, 0.2, 1.7, 0.3));
        }
    }

    return {duration: DURATION, keyframes: [1.2, 4.6, 6.8, 12.6], render};
}
