// Figure for "Interfaces instead of port lists": the three signals of each ready/valid connection of the example fold
// into one interface connection, then values flow through the two registers, with ready going backwards.

import {createSvg, el, lerp, progress, set, setClass, setText, visibility} from "./util.js";

const WIDTH = 420;
const HEIGHT = 200;
const DURATION = 18;

const WIRE_Y = 88;
const LANES = [{name: "data", forward: true}, {name: "valid", forward: true}, {name: "ready", forward: false}];
const LANE_Y_OPEN = [WIRE_Y - 20, WIRE_Y, WIRE_Y + 20];
const LANE_Y_FOLDED = [WIRE_Y - 6, WIRE_Y, WIRE_Y + 6];

const BOX_TOP = WIRE_Y - 36;
const BOX_BOTTOM = WIRE_Y + 36;
const LEFT_BOX = [90, 170];
const RIGHT_BOX = [250, 330];
// the connections of TwoStages: its input port, the link wire and its output port
const SEGMENTS = [{x0: 0, x1: 90, name: "input"}, {x0: 170, x1: 250, name: "link"}, {x0: 330, x1: 420, name: "output"}];

const FOLD_START = 4.2;
const FOLD_END = 5.8;
const TRAFFIC_START = 6.6;
const CYCLE = 1.3;
const DOWNSTREAM_READY = [true, true, true, false, false, true, true, true];

const X_SOURCE = -20;
const X_LEFT = (LEFT_BOX[0] + LEFT_BOX[1]) / 2;
const X_RIGHT = (RIGHT_BOX[0] + RIGHT_BOX[1]) / 2;
const X_SINK = 440;

// Cycles of the two registers of the example, with a source that always has data.
// Each register takes a new value when its output is empty or being taken (`input.ready = output.ready || !output.valid`).
function simulate() {
    let left = null;
    let right = null;
    let next = 1;
    return DOWNSTREAM_READY.map(ready => {
        const rightFree = right === null || ready;
        const leftFree = left === null || rightFree;
        const cycle = {left, right, ready, rightFree, leftFree, incoming: leftFree ? next++ : null};
        right = rightFree ? left : right;
        left = leftFree ? cycle.incoming : left;
        return cycle;
    });
}

const CYCLES = simulate();

function createChip(parent) {
    const g = el(parent, "g");
    el(g, "rect", {x: -15, y: WIRE_Y - 11, width: 30, height: 22, rx: 4, class: "v-chip"});
    const text = el(g, "text", {y: WIRE_Y + 5, "text-anchor": "middle", class: "v-strong v-mono", "font-size": 13});
    return {g, text};
}

export function createInterfacesViz(container) {
    const svg = createSvg(container, WIDTH, HEIGHT,
        "Two Register modules connected in a row. First every connection is three separate wires: data and valid " +
        "forward, ready backward. Then each folds into one interface connection, named input, link and output like in " +
        "the code. Values flow through the registers and wait when downstream is not ready.");
    const content = el(svg, "g");

    // per connection: the three signals, the cable they fold into, and its name
    const segments = SEGMENTS.map(({x0, x1, name}) => {
        const cable = el(content, "rect", {x: x0 - 4, y: WIRE_Y - 12, width: x1 - x0 + 8, height: 24, rx: 6, class: "v-cable"});
        const lanes = LANES.map(lane => ({
            path: el(content, "path", {class: "v-lane"}),
            arrow: el(content, "path", {class: "v-lane"}),
            forward: lane.forward,
        }));
        const label = el(content, "text", {x: (x0 + x1) / 2, y: WIRE_Y - 20, "text-anchor": "middle", class: "v-strong v-mono", "font-size": 12}, name);
        return {x0, x1, cable, lanes, label};
    });
    const laneLabels = LANES.map(lane => el(content, "text", {x: 210, "text-anchor": "middle", class: "v-muted v-mono", "font-size": 10}, lane.name));

    for (const [x0, x1] of [LEFT_BOX, RIGHT_BOX]) {
        el(content, "rect", {x: x0, y: BOX_TOP, width: x1 - x0, height: BOX_BOTTOM - BOX_TOP, rx: 4, class: "v-box"});
        el(content, "path", {d: `M${x0},${BOX_BOTTOM - 18} l7,5 l-7,5`, class: "v-wire"});
        el(content, "text", {x: (x0 + x1) / 2, y: BOX_TOP - 8, "text-anchor": "middle", class: "v-muted v-mono", "font-size": 12}, "Register");
    }
    const type = el(content, "text", {x: 210, y: BOX_BOTTOM + 22, "text-anchor": "middle", class: "v-muted v-mono", "font-size": 12},
        "interface Stream(uint(8))");

    const chips = Array.from({length: 3}, () => createChip(content));
    const status = el(content, "text", {x: 12, y: HEIGHT - 10, class: "v-text", "font-size": 14});

    function render(t) {
        const fold = progress(t, FOLD_START, FOLD_END);
        const traffic = t >= TRAFFIC_START;
        const k = Math.min(CYCLES.length - 1, Math.max(0, Math.floor((t - TRAFFIC_START) / CYCLE)));
        const u = t - TRAFFIC_START - k * CYCLE;
        const c = CYCLES[k];

        // which signals are active during the traffic: valid when a value is offered, ready when it can be taken
        const active = [
            [true, true, c.leftFree],
            [c.left !== null, c.left !== null, c.rightFree],
            [c.right !== null, c.right !== null, c.ready],
        ];
        segments.forEach((segment, s) => {
            set(segment.cable, "opacity", fold);
            set(segment.label, "opacity", fold);
            segment.lanes.forEach((lane, i) => {
                const y = lerp(LANE_Y_OPEN[i], LANE_Y_FOLDED[i], fold);
                set(lane.path, "d", `M${segment.x0},${y} H${segment.x1}`);
                const mid = (segment.x0 + segment.x1) / 2 + (lane.forward ? 3 : -3);
                const dx = lane.forward ? -4 : 4;
                set(lane.arrow, "d", `M${mid + dx},${y - 3.5} L${mid},${y} L${mid + dx},${y + 3.5}`);
                const on = traffic && active[s][i];
                setClass(lane.path, on ? "v-lane on" : "v-lane");
                setClass(lane.arrow, on ? "v-lane on" : "v-lane");
            });
        });
        laneLabels.forEach((label, i) => {
            set(label, "y", LANE_Y_OPEN[i] - 4);
            set(label, "opacity", 1 - progress(t, FOLD_START - 0.4, FOLD_START + 0.2));
        });
        set(type, "opacity", fold);

        // values move on the clock edge in the middle of each cycle
        const move = traffic ? progress(u, 0.3, 0.85) : 0;
        let used = 0;
        const chip = (x, value, opacity) => {
            const {g, text} = chips[used++];
            set(g, "transform", `translate(${x.toFixed(2)} 0)`);
            set(g, "opacity", opacity);
            setText(text, String(value));
        };
        if (traffic) {
            if (c.right !== null) {
                chip(c.ready ? lerp(X_RIGHT, X_SINK, move) : X_RIGHT, c.right, c.ready ? 1 - move : 1);
            }
            if (c.left !== null) {
                chip(c.rightFree ? lerp(X_LEFT, X_RIGHT, move) : X_LEFT, c.left, 1);
            }
            if (c.incoming !== null) {
                chip(lerp(X_SOURCE, X_LEFT, move), c.incoming, move);
            }
        }
        for (let i = used; i < chips.length; i++) {
            set(chips[i].g, "opacity", 0);
        }

        let text;
        if (t < FOLD_START) {
            text = "Without interfaces: three signals per connection";
        } else if (!traffic) {
            text = "With an interface: one port, one connection";
        } else if (!c.ready) {
            text = "ready is low, so both registers hold their values";
        } else {
            text = "data and valid go forward, ready comes back";
        }
        setText(status, text);

        set(content, "opacity", visibility(t, 0, DURATION, 0.5));
    }

    return {duration: DURATION, keyframes: [3.6, 6.2, TRAFFIC_START + 1.2, TRAFFIC_START + 3 * CYCLE + 1.2, TRAFFIC_START + 5 * CYCLE + 1.2], render};
}
