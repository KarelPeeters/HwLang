// Plays the animated figures. Each figure is a pure function of time, `render(t)`, so it can be paused, stepped
// between key frames and shown as a still image. Figures only animate while they are on screen.
// With reduced motion, figures start paused and the step button moves between key frames.
// A figure gives its key frames as a list, or as a `nextKeyframe(t)` function when it runs forever.

import {createRangesViz} from "./viz/ranges.js";
import {createDomainsViz} from "./viz/domains.js";
import {createSkidViz} from "./viz/skid.js";
import {createInterfacesViz} from "./viz/interfaces.js";
import {createGenericsViz} from "./viz/generics.js";

const FACTORIES = {
    ranges: createRangesViz,
    domains: createDomainsViz,
    skid: createSkidViz,
    interfaces: createInterfacesViz,
    generics: createGenericsViz,
};

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const ICON_PAUSE = `<svg viewBox="0 0 14 14" aria-hidden="true"><rect x="3" y="2" width="3" height="10" fill="currentColor"/><rect x="8" y="2" width="3" height="10" fill="currentColor"/></svg>`;
const ICON_PLAY = `<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M4 2 L12 7 L4 12 Z" fill="currentColor"/></svg>`;
const ICON_STEP = `<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M2 2 L9 7 L2 12 Z" fill="currentColor"/><rect x="10" y="2" width="2" height="10" fill="currentColor"/></svg>`;

class Player {
    constructor(figure, factory) {
        const frame = document.createElement("div");
        frame.className = "viz-frame";
        const footer = document.createElement("div");
        footer.className = "viz-footer";
        const caption = figure.querySelector("figcaption");
        if (caption) footer.append(caption);
        figure.prepend(frame, footer);

        // figures with their own controls can pause the animation when the viewer takes over
        this.viz = factory(frame, {pause: () => this.setPlaying(false)});
        this.visible = false;
        this.playing = !reducedMotion.matches;
        this.t = this.playing ? 0 : this.nextKeyframe(-1);
        this.lastFrame = null;

        const controls = document.createElement("div");
        controls.className = "viz-controls";
        this.playButton = document.createElement("button");
        this.playButton.type = "button";
        this.playButton.addEventListener("click", () => this.setPlaying(!this.playing));
        const stepButton = document.createElement("button");
        stepButton.type = "button";
        stepButton.innerHTML = ICON_STEP;
        stepButton.setAttribute("aria-label", "Next step");
        stepButton.title = "Next step";
        stepButton.addEventListener("click", () => this.step());
        controls.append(this.playButton, stepButton);
        footer.append(controls);

        this.updateButton();
        this.viz.render(this.t);

        reducedMotion.addEventListener("change", () => this.setPlaying(!reducedMotion.matches));
        new IntersectionObserver(entries => {
            this.visible = entries[entries.length - 1].isIntersecting;
            this.schedule();
        }, {threshold: 0.2}).observe(frame);
    }

    setPlaying(playing) {
        this.playing = playing;
        this.updateButton();
        this.schedule();
    }

    updateButton() {
        this.playButton.innerHTML = this.playing ? ICON_PAUSE : ICON_PLAY;
        const label = this.playing ? "Pause animation" : "Play animation";
        this.playButton.setAttribute("aria-label", label);
        this.playButton.title = label;
    }

    nextKeyframe(t) {
        if (this.viz.nextKeyframe) {
            return this.viz.nextKeyframe(t);
        }
        const keyframes = this.viz.keyframes;
        return keyframes.find(k => k > t + 1e-3) ?? keyframes[0];
    }

    step() {
        this.setPlaying(false);
        this.t = this.nextKeyframe(this.t);
        this.viz.render(this.t);
    }

    schedule() {
        if (this.playing && this.visible && this.lastFrame === null) {
            this.lastFrame = performance.now();
            requestAnimationFrame(now => this.frame(now));
        }
    }

    frame(now) {
        if (!this.playing || !this.visible) {
            this.lastFrame = null;
            return;
        }
        // cap the step, so returning to a background tab doesn't skip ahead, the first frame time can be slightly
        // before the time we started playing
        const dt = Math.min(Math.max(now - this.lastFrame, 0), 100) / 1000;
        this.lastFrame = now;
        this.t = (this.t + dt) % this.viz.duration;
        this.viz.render(this.t);
        requestAnimationFrame(now => this.frame(now));
    }
}

for (const figure of document.querySelectorAll("figure.viz[data-viz]")) {
    const factory = FACTORIES[figure.dataset.viz];
    if (factory) {
        new Player(figure, factory);
    }
}
