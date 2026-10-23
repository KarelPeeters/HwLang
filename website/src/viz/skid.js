// Figure for "Code runs top to bottom": the ready/valid example from the page, with random stalls upstream and
// downstream. At every clock edge, the four steps of the process run in order (shown in slow motion), and each step
// sees what the previous steps did.

import {createSvg, el, lerp, progress, set, setClass, setText} from "./util.js";

const WIDTH = 420;
const HEIGHT = 256;

const CYCLE = 5.0;
const ENV_END = 0.7;
const STEP = 1.0;
const SEED = 78;
const P_VALID = 0.7;
const P_READY = 0.55;

const LINE_Y = [36, 58, 80, 102];
const BOX_Y = 140;
const BOX_HEIGHT = 44;
const WIRE_Y = BOX_Y + BOX_HEIGHT / 2;
const X_ENTER = -24;
const X_INPUT = 52;
const X_SKID = 170;
const X_OUTPUT = 354;
const X_ADD = (X_SKID + X_OUTPUT) / 2;
const REG_WIDTH = 76;
const X_GONE = 446;
const CHIP_WIDTH = 34;

const LINES = [
    [["if", "tk-kw"], [" (input_valid && input_ready)", ""]],
    [["if", "tk-kw"], [" (output_valid && output_ready)", ""]],
    [["if", "tk-kw"], [" (skid_valid && !output_valid)", ""]],
    [["input_ready = !skid_valid;", ""]],
];

function mulberry32(seed) {
    return () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Cycle-level model of the example, with a random upstream and downstream. Values are numbered in order.
// Each cycle records the state before the clock edge and which of the three if statements were taken.
export class SkidModel {
    constructor(seed = SEED) {
        this.random = mulberry32(seed);
        this.state = {offered: null, next: 1, inputReady: true, skid: null, output: null};
        this.cycles = [];
    }

    cycle(k) {
        while (this.cycles.length <= k) {
            this.advance();
        }
        return this.cycles[k];
    }

    advance() {
        const s = this.state;
        // upstream offers a new value (and keeps offering it until accepted), downstream is randomly ready
        const fresh = s.offered === null && this.random() < P_VALID;
        if (fresh) {
            s.offered = s.next++;
        }
        const outputReady = this.random() < P_READY;
        const before = {offered: s.offered, fresh, inputReady: s.inputReady, skid: s.skid, output: s.output, outputReady};

        let {offered, skid, output} = s;
        const accepted = offered !== null && s.inputReady;
        if (accepted) {
            skid = offered;
            offered = null;
        }
        const taken = output !== null && outputReady;
        if (taken) {
            output = null;
        }
        const skidAfterInput = skid;
        const moved = skid !== null && output === null;
        if (moved) {
            output = skid;
            skid = null;
        }
        const inputReady = skid === null;

        this.cycles.push({before, accepted, taken, moved, skidAfterInput, after: {skid, output, inputReady}});
        Object.assign(s, {offered, skid, output, inputReady});
    }
}

function stepStart(i) {
    return ENV_END + i * STEP;
}

function statusText(c, step) {
    const b = c.before;
    if (step < 0) {
        const upstream = b.offered === null ? "Upstream has nothing to send"
            : b.fresh ? `Upstream offers ${b.offered}` : `${b.offered} is still waiting at the input`;
        return `${upstream}, downstream ${b.outputReady ? "is ready" : "is not ready"}`;
    }
    switch (step) {
        case 0:
            if (c.accepted) return `Input handshake: ${b.offered} goes into the skid register`;
            return b.offered === null ? "No input to accept" : `input_ready is low, so ${b.offered} waits`;
        case 1:
            if (c.taken) return `Output handshake: ${b.output + 5} is taken downstream`;
            return b.output === null ? "The output is empty" : `output_ready is low, so ${b.output + 5} stays`;
        case 2:
            if (c.moved) return `The output is free: ${c.skidAfterInput} + 5 = ${c.skidAfterInput + 5} moves to it`;
            return c.skidAfterInput === null ? "The skid register is empty"
                : `The output is still full, so ${c.skidAfterInput} stays in skid`;
        default:
            return c.after.inputReady ? "skid is empty, so input_ready goes high" : "skid is full, so input_ready goes low";
    }
}

function createChip(parent) {
    const g = el(parent, "g");
    const rect = el(g, "rect", {x: -CHIP_WIDTH / 2, y: WIRE_Y - 12, width: CHIP_WIDTH, height: 24, rx: 4});
    const text = el(g, "text", {y: WIRE_Y + 5, "text-anchor": "middle", class: "v-strong v-mono", "font-size": 14});
    return {g, rect, text};
}

function drawChip(chip, x, value, opacity, processed) {
    set(chip.g, "transform", `translate(${x.toFixed(2)} 0)`);
    set(chip.g, "opacity", opacity);
    setClass(chip.rect, processed ? "v-chip-done" : "v-chip");
    setText(chip.text, String(value));
}

function createLamp(parent, x, label) {
    const circle = el(parent, "circle", {cx: x, cy: BOX_Y + BOX_HEIGHT + 16, r: 5});
    el(parent, "text", {x, y: BOX_Y + BOX_HEIGHT + 34, "text-anchor": "middle", class: "v-muted v-mono", "font-size": 12}, label);
    return circle;
}

export function createSkidViz(container) {
    const model = new SkidModel();
    const svg = createSvg(container, WIDTH, HEIGHT,
        "Values move from the input port to a skid register and on to the output register. Upstream and downstream " +
        "stall at random. At every clock edge, four steps run in order: accept input into the skid register, clear " +
        "the output once it is taken, move the skid value plus five to the output if it is free, and set input_ready.");

    // the steps of the process
    el(svg, "text", {x: 12, y: 14, class: "v-muted", "font-size": 12}, "At every clock edge, in order:");
    const lines = LINES.map((segments, i) => {
        const highlight = el(svg, "rect", {x: 8, y: LINE_Y[i] - 15, width: 404, height: 21, rx: 4, class: "v-hl", opacity: 0});
        const text = el(svg, "text", {x: 16, y: LINE_Y[i], class: "v-strong v-mono", "font-size": 13});
        for (const [str, cls] of segments) {
            el(text, "tspan", cls ? {class: cls} : {}, str);
        }
        const mark = el(svg, "text", {x: 400, y: LINE_Y[i], "text-anchor": "end", class: "v-mono", "font-size": 13});
        return {highlight, text, mark};
    });

    // input port, skid register and output register
    const skidLeft = X_SKID - REG_WIDTH / 2;
    const outputLeft = X_OUTPUT - REG_WIDTH / 2;
    el(svg, "path", {
        d: `M0,${WIRE_Y} H14 M98,${WIRE_Y} H${skidLeft} M${skidLeft + REG_WIDTH},${WIRE_Y} H${outputLeft} ` +
            `M${outputLeft + REG_WIDTH},${WIRE_Y} H416 m-6,-5 l6,5 l-6,5`,
        class: "v-wire",
    });
    el(svg, "path", {d: `M14,${BOX_Y} H86 L98,${WIRE_Y} L86,${BOX_Y + BOX_HEIGHT} H14 Z`, class: "v-port"});
    for (const x of [skidLeft, outputLeft]) {
        el(svg, "rect", {x, y: BOX_Y, width: REG_WIDTH, height: BOX_HEIGHT, rx: 4, class: "v-box"});
        el(svg, "path", {d: `M${x},${BOX_Y + BOX_HEIGHT - 16} l7,5 l-7,5`, class: "v-wire"});
    }
    for (const [x, label] of [[X_INPUT, "input"], [X_SKID, "skid"], [X_OUTPUT, "output"]]) {
        el(svg, "text", {x, y: BOX_Y - 8, "text-anchor": "middle", class: "v-muted v-mono", "font-size": 13}, label);
    }
    const lamps = {
        inputValid: createLamp(svg, 30, "valid"),
        inputReady: createLamp(svg, 76, "ready"),
        skidValid: createLamp(svg, X_SKID, "valid"),
        outputValid: createLamp(svg, X_OUTPUT - 24, "valid"),
        outputReady: createLamp(svg, X_OUTPUT + 24, "ready"),
    };

    const chips = Array.from({length: 4}, () => createChip(svg));

    // the addition, drawn over the values so they pass through it
    const adder = el(svg, "rect", {x: X_ADD - 20, y: WIRE_Y - 15, width: 40, height: 30, rx: 4, class: "v-op"});
    el(svg, "text", {x: X_ADD, y: WIRE_Y + 5, "text-anchor": "middle", class: "v-strong v-mono", "font-size": 14}, "+5");
    const status = el(svg, "text", {x: 12, y: HEIGHT - 6, class: "v-text", "font-size": 14});

    function render(t) {
        const k = Math.floor(t / CYCLE);
        const u = t - k * CYCLE;
        const c = model.cycle(k);
        const b = c.before;
        const step = u < ENV_END ? -1 : Math.min(3, Math.floor((u - ENV_END) / STEP));
        // progress of the movement in step i, and whether its result is shown yet
        const move = i => progress(u, stepStart(i) + 0.3, stepStart(i) + 0.8);
        const decided = i => u >= stepStart(i) + 0.3;

        // code lines: highlight the current step, mark the result of each evaluated step
        const taken = [c.accepted, c.taken, c.moved];
        lines.forEach((line, i) => {
            set(line.highlight, "opacity", i === step ? 1 : 0);
            set(line.text, "opacity", step === -1 || i <= step ? 1 : 0.55);
            if (!decided(i)) {
                setText(line.mark, "");
            } else if (i < 3) {
                setText(line.mark, taken[i] ? "✓" : "–");
                setClass(line.mark, taken[i] ? "v-mono v-green" : "v-mono v-muted");
            } else {
                setText(line.mark, c.after.inputReady ? "= 1" : "= 0");
                setClass(line.mark, "v-mono v-muted");
            }
        });

        // values: the one offered at the input, the one in the skid register, and the one in the output register
        let used = 0;
        const chip = (x, value, opacity, processed) => drawChip(chips[used++], x, value, opacity, processed);
        if (b.offered !== null) {
            const arrive = b.fresh ? progress(u, 0.05, 0.6) : 1;
            let x = lerp(X_ENTER, X_INPUT, arrive);
            let processed = false;
            if (c.accepted) {
                x = lerp(x, X_SKID, move(0));
                if (c.moved && b.skid === null) {
                    x = lerp(x, X_OUTPUT, move(2));
                    processed = move(2) >= 0.5;
                }
            }
            chip(x, processed ? b.offered + 5 : b.offered, arrive, processed);
        }
        if (b.skid !== null) {
            const m = c.moved ? move(2) : 0;
            chip(lerp(X_SKID, X_OUTPUT, m), m >= 0.5 ? b.skid + 5 : b.skid, 1, m >= 0.5);
        }
        if (b.output !== null) {
            const m = c.taken ? move(1) : 0;
            chip(lerp(X_OUTPUT, X_GONE, m), b.output + 5, 1 - m, true);
        }
        for (let i = used; i < chips.length; i++) {
            set(chips[i].g, "opacity", 0);
        }
        const passing = c.moved && Math.abs(lerp(X_SKID, X_OUTPUT, move(2)) - X_ADD) < 30;
        setClass(adder, passing ? "v-op on" : "v-op");

        // handshake signals, following the steps
        const on = (lamp, value) => setClass(lamp, value ? "v-lamp on" : "v-lamp");
        on(lamps.inputValid, b.offered !== null && !(c.accepted && decided(0)));
        on(lamps.inputReady, decided(3) ? c.after.inputReady : b.inputReady);
        on(lamps.skidValid, c.moved && decided(2) ? false : c.accepted && decided(0) ? true : b.skid !== null);
        const outputValid = c.moved && decided(2) ? true : c.taken && decided(1) ? false : b.output !== null;
        on(lamps.outputValid, outputValid);
        on(lamps.outputReady, b.outputReady);

        setText(status, statusText(c, step));
        set(status, "opacity", progress(u - (step < 0 ? 0 : stepStart(step)), 0, 0.15));
    }

    // key frames: after the inputs change, and after each step
    function nextKeyframe(t) {
        const k = Math.max(0, Math.floor(t / CYCLE));
        for (let cycle = k; ; cycle++) {
            const times = [ENV_END - 0.05, ...[0, 1, 2, 3].map(i => stepStart(i) + 0.9)].map(x => cycle * CYCLE + x);
            const next = times.find(x => x > t + 1e-3);
            if (next !== undefined) return next;
        }
    }

    return {duration: Infinity, nextKeyframe, render};
}
