// Figure for "Generics are ordinary code": the module of the example for a given number of inputs, with its numbered
// ports, the chain of adders that each grow one bit wider, and the range of `total`. It cycles through the values on
// its own; moving the slider pauses that and shows the chosen value. Changes animate with CSS transitions, so they
// also work while the player is paused.

import {createSvg, el, setText} from "./util.js";

const WIDTH = 420;
const HEIGHT = 262;
const MAX_INPUTS = 6;
const SEQUENCE = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2];
const STEP_TIME = 2.2;

const BOX_X0 = 92;
const BOX_X1 = 300;
const BOX_TOP = 14;
const BOX_BOTTOM = 222;
const PORT_Y = i => 36 + i * 34;
// adder j adds value_(j + 1) to the sum of the previous values
const ADDER_X = j => 140 + j * 30;
const ADDER_R = 9;

function sumBits(values) {
    return Math.ceil(Math.log2(255 * values + 1));
}

export function createGenericsViz(container, controls) {
    // the slider, above the figure
    const row = document.createElement("label");
    row.className = "viz-slider";
    const name = document.createElement("code");
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "1";
    slider.max = String(MAX_INPUTS);
    slider.step = "1";
    row.append(name, slider);

    const svg = createSvg(container, WIDTH, HEIGHT,
        "The Sum module for the chosen number of inputs: input ports value_0 up to value_(inputs - 1), a chain of " +
        "adders that each produce a result one bit wider when needed, and the output total with a range that grows " +
        "with the number of inputs.");
    container.insertBefore(row, svg);
    el(svg, "rect", {x: BOX_X0, y: BOX_TOP, width: BOX_X1 - BOX_X0, height: BOX_BOTTOM - BOX_TOP, rx: 6, class: "v-box"});
    const title = el(svg, "text", {x: BOX_X1 - 8, y: BOX_TOP + 16, "text-anchor": "end", class: "v-muted v-mono", "font-size": 12});

    // input ports, all of them exist and are shown or hidden
    const ports = Array.from({length: MAX_INPUTS}, (_, i) => {
        const g = el(svg, "g", {class: "v-anim"});
        el(g, "text", {x: BOX_X0 - 8, y: PORT_Y(i) + 4, "text-anchor": "end", class: "v-strong v-mono", "font-size": 12}, `value_${i}`);
        // value_0 runs to above the first adder, the others into the side of their adder
        const end = i === 0 ? ADDER_X(0) : ADDER_X(i - 1) - ADDER_R;
        el(g, "path", {d: `M${BOX_X0 - 4},${PORT_Y(i)} H${end}`, class: "v-wire"});
        return g;
    });
    // with a single input, value_0 goes straight to the output
    const direct = el(svg, "path", {d: `M${ADDER_X(0)},${PORT_Y(0)} H${BOX_X1}`, class: "v-wire v-anim"});
    const firstDrop = el(svg, "path", {d: `M${ADDER_X(0)},${PORT_Y(0)} V${PORT_Y(1) - ADDER_R}`, class: "v-wire v-anim"});

    // adders: the sum comes in from above, the next value from the left
    const adders = Array.from({length: MAX_INPUTS - 1}, (_, j) => {
        const x = ADDER_X(j);
        const y = PORT_Y(j + 1);
        const g = el(svg, "g", {class: "v-anim"});
        el(g, "circle", {cx: x, cy: y, r: ADDER_R, class: "v-op"});
        el(g, "text", {x, y: y + 5, "text-anchor": "middle", class: "v-strong v-mono", "font-size": 14}, "+");
        el(g, "text", {x: x + ADDER_R + 3, y: y - 5, class: "v-text-a v-mono", "font-size": 11}, `${sumBits(j + 2)}b`);
        // to the next adder, or to the output when this is the last one
        const next = el(svg, "path", {d: `M${x + ADDER_R},${y} H${ADDER_X(j + 1)} V${PORT_Y(j + 2) - ADDER_R}`, class: "v-wire v-anim"});
        const last = el(svg, "path", {d: `M${x + ADDER_R},${y} H${BOX_X1}`, class: "v-wire v-anim"});
        return {g, next, last};
    });

    // the output port follows the last adder
    const output = el(svg, "g", {class: "v-anim"});
    el(output, "path", {d: `M${BOX_X1},0 H${BOX_X1 + 14} m-6,-5 l6,5 l-6,5`, class: "v-wire"});
    el(output, "text", {x: BOX_X1 + 20, y: 4, class: "v-strong v-mono", "font-size": 12}, "total");
    const range = el(output, "text", {x: BOX_X1 + 20, y: 20, class: "v-text-a v-mono", "font-size": 11});

    const status = el(svg, "text", {x: 12, y: HEIGHT - 8, class: "v-text", "font-size": 14});

    const show = (node, visible) => {
        node.style.opacity = visible ? "1" : "0";
    };

    let shown = null;

    function setInputs(n) {
        if (n === shown) return;
        shown = n;
        slider.value = String(n);
        setText(name, `inputs = ${n}`);
        setText(title, `Sum(inputs=${n})`);
        ports.forEach((port, i) => show(port, i < n));
        show(direct, n === 1);
        show(firstDrop, n > 1);
        adders.forEach((adder, j) => {
            show(adder.g, j < n - 1);
            show(adder.next, j < n - 2);
            show(adder.last, j === n - 2);
        });
        const y = n === 1 ? PORT_Y(0) : PORT_Y(n - 1);
        output.style.transform = `translate(0px, ${y}px)`;
        setText(range, `uint(0..=${255 * n})`);
        const portList = n === 1 ? "value_0" : n === 2 ? "value_0, value_1" : `value_0 … value_${n - 1}`;
        setText(status, `${portList}, total has ${sumBits(n)} bits`);
    }

    slider.addEventListener("input", () => {
        controls.pause();
        setInputs(Number(slider.value));
    });

    function render(t) {
        setInputs(SEQUENCE[Math.floor(t / STEP_TIME) % SEQUENCE.length]);
    }

    return {
        duration: SEQUENCE.length * STEP_TIME,
        keyframes: SEQUENCE.map((_, i) => i * STEP_TIME + 0.6),
        render,
    };
}
