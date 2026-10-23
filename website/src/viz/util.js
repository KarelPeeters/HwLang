// Small helpers shared by the figures.

const SVG_NS = "http://www.w3.org/2000/svg";

// Create an SVG element with the given attributes and append it to `parent`.
export function el(parent, tag, attrs = {}, text = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
        node.setAttribute(key, value);
    }
    if (text !== null) {
        node.textContent = text;
    }
    parent?.append(node);
    return node;
}

export function createSvg(container, width, height, label) {
    const svg = el(null, "svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label});
    container.prepend(svg);
    return svg;
}

export function clamp01(x) {
    return Math.min(1, Math.max(0, x));
}

export function lerp(a, b, k) {
    return a + (b - a) * k;
}

// Eased progress of `t` through the interval `[start, end]`, 0 before and 1 after.
export function progress(t, start, end) {
    const x = clamp01((t - start) / (end - start));
    return x * x * (3 - 2 * x);
}

// Opacity for something that fades in at `start` and out at `end`.
export function visibility(t, start, end, fade = 0.4) {
    return Math.min(progress(t, start, start + fade), 1 - progress(t, end - fade, end));
}

// Set an attribute only when its value changed, to avoid needless style recalculation.
export function set(node, key, value) {
    const str = typeof value === "number" ? value.toFixed(2) : value;
    if (node.getAttribute(key) !== str) {
        node.setAttribute(key, str);
    }
}

export function setText(node, text) {
    if (node.textContent !== text) {
        node.textContent = text;
    }
}

export function setClass(node, cls) {
    set(node, "class", cls);
}
