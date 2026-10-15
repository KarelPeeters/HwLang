// Core math, easing, colour and timing helpers. Everything is a pure function of time.

export const W = 1920;
export const H = 1080;

export const C = {
  bg: "#0a0c11",
  bg2: "#10131a",
  panel: "#131722",
  rule: "#232938",
  grid: "#1a1f2b",
  text: "#e9edf4",
  muted: "#6b7385",
  dim: "#3a4152",
  cyan: "#3ce0ff",
  magenta: "#ff4fd8",
  amber: "#ffb23e",
  red: "#ff3b5c",
  green: "#4cf2a0",
  violet: "#9b7bff",
};

export const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const invlerp = (a, b, x) => clamp((x - a) / (b - a));
export const remap = (x, a, b, c, d) => lerp(c, d, invlerp(a, b, x));
export const smooth = (t) => t * t * (3 - 2 * t);

export const ease = {
  linear: (t) => t,
  inQuad: (t) => t * t,
  outQuad: (t) => 1 - (1 - t) * (1 - t),
  inOutQuad: (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2),
  outCubic: (t) => 1 - (1 - t) ** 3,
  inCubic: (t) => t * t * t,
  inOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
  outQuart: (t) => 1 - (1 - t) ** 4,
  outExpo: (t) => (t >= 1 ? 1 : 1 - 2 ** (-10 * t)),
  inExpo: (t) => (t <= 0 ? 0 : 2 ** (10 * t - 10)),
  inOutExpo: (t) =>
    t <= 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? 2 ** (20 * t - 10) / 2 : (2 - 2 ** (-20 * t + 10)) / 2,
  outBack: (t, s = 1.70158) => 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2,
  outElastic: (t) =>
    t <= 0 ? 0 : t >= 1 ? 1 : 2 ** (-10 * t) * Math.sin((t * 10 - 0.75) * ((2 * Math.PI) / 3)) + 1,
};

// progress of t through [a, a+d], eased
export const prog = (t, a, d, e = ease.outCubic) => {
  const x = clamp((t - a) / d);
  return x <= 0 ? 0 : x >= 1 ? 1 : e(x);
};

// window: rises over `fi` after a, holds, falls over `fo` before b
export function win(t, a, b, fi = 0.3, fo = 0.3) {
  if (t < a || t > b) return 0;
  return Math.min(fi > 0 ? clamp((t - a) / fi) : 1, fo > 0 ? clamp((b - t) / fo) : 1);
}

// deterministic hash-based randomness
export function hash(n) {
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
  n = n ^ (n >>> 16);
  return (n >>> 0) / 4294967296;
}
export const rnd = (i, j = 0, k = 0) => hash((i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ 0x9e3779b9);

export function noise1(x, seed = 0) {
  const i = Math.floor(x);
  const f = x - i;
  return lerp(rnd(i, seed), rnd(i + 1, seed), smooth(f));
}

// typing / reveal helpers
export function typed(str, t, start, cps) {
  const n = Math.floor(clamp((t - start) * cps, 0, str.length));
  return str.slice(0, n);
}

export function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export function mixHex(h1, h2, t) {
  const a = parseInt(h1.slice(1), 16);
  const b = parseInt(h2.slice(1), 16);
  const r = Math.round(lerp((a >> 16) & 255, (b >> 16) & 255, t));
  const g = Math.round(lerp((a >> 8) & 255, (b >> 8) & 255, t));
  const bl = Math.round(lerp(a & 255, b & 255, t));
  return `#${((1 << 24) | (r << 16) | (g << 8) | bl).toString(16).slice(1)}`;
}

// ---- audio-derived timing ----
export const A = { data: null };

export function setAudio(data) {
  A.data = data;
}

export function env(name, t) {
  const d = A.data;
  if (!d) return 0;
  const arr = d.env[name];
  const x = t * d.env_fps;
  const i = Math.floor(x);
  if (i < 0 || i + 1 >= arr.length) return 0;
  return lerp(arr[i], arr[i + 1], x - i);
}

// index of last beat <= t and phase within it
export function beatInfo(t) {
  const b = A.data ? A.data.beats : [];
  let lo = 0;
  let hi = b.length - 1;
  if (!b.length || t < b[0]) return { i: -1, phase: 0, since: 1e9, len: 0.4644 };
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (b[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  const len = lo + 1 < b.length ? b[lo + 1] - b[lo] : 0.4644;
  return { i: lo, phase: (t - b[lo]) / len, since: t - b[lo], len };
}

// sharp decaying pulse on each beat
export function pulse(t, decay = 6) {
  const { since } = beatInfo(t);
  return Math.exp(-since * decay);
}

export function kick(t) {
  return clamp(env("low", t) * 1.1);
}
