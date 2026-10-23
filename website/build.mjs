// Builds the landing page into a static directory (default `website/dist`, override with `--out <dir>`).
//
// The page template `src/index.html` refers to the code examples in `examples/`:
// * `<pre data-example="name.kh"></pre>` is replaced by the highlighted example,
// * `<pre data-diagnostic="name.kh"></pre>` by the compiler output stored in `name.err`,
// * `data-demo="name.kh"` on a link becomes an `href` that opens the full example in the web demo.
// Everything after the hidden marker line in an example is only included in the demo link, not shown on the page.
// The examples and diagnostics are checked against the compiler by `test_website_examples.py`.
//
// No dependencies, highlighting uses the TextMate grammar of the VSCode extension.

import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import {fileURLToPath} from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const SRC_DIR = path.join(ROOT, "src");
const EXAMPLES_DIR = path.join(ROOT, "examples");
const GRAMMAR_PATH = path.join(ROOT, "..", "lsp_client", "syntaxes", "khdl.tmLanguage.json");
const HIDDEN_MARKER = "// ---- website: hidden below this line ----";

function parseArgs(argv) {
    let out = path.join(ROOT, "dist");
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === "--out") {
            out = path.resolve(argv[++i]);
        } else {
            throw new Error(`unknown argument ${argv[i]}`);
        }
    }
    return {out};
}

function escapeHtml(s) {
    return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

// Minimal TextMate engine, supporting the subset used by the KHDL grammar:
// `match` rules, `begin`/`end` rules with nested patterns and capture 0 names, and `$self`/`#repository` includes.
class Grammar {
    constructor(json) {
        this.repository = {};
        for (const [name, entry] of Object.entries(json.repository ?? {})) {
            this.repository[name] = this.compileList(entry.patterns);
        }
        this.patterns = this.compileList(json.patterns);
    }

    compileList(patterns) {
        return (patterns ?? []).map(p => {
            if (p.include !== undefined) {
                return {include: p.include};
            }
            if (p.match !== undefined) {
                return {name: p.name, re: new RegExp(p.match, "gm")};
            }
            return {
                name: p.name,
                begin: new RegExp(p.begin, "gm"),
                end: new RegExp(p.end, "gm"),
                beginName: p.beginCaptures?.[0]?.name,
                endName: p.endCaptures?.[0]?.name,
                patterns: this.compileList(p.patterns),
            };
        });
    }

    flatten(list) {
        return list.flatMap(rule => {
            if (rule.include === undefined) return [rule];
            if (rule.include === "$self") return this.flatten(this.patterns);
            return this.flatten(this.repository[rule.include.slice(1)]);
        });
    }

    // Returns a list of `{text, scope}` tokens, where scope is the innermost scope name or undefined.
    tokenize(text) {
        const tokens = [];
        const emit = (str, scope) => {
            if (str.length > 0) tokens.push({text: str, scope});
        };

        const run = (pos, patterns, end, scope) => {
            const rules = this.flatten(patterns);
            while (pos < text.length) {
                let endMatch = null;
                if (end) {
                    end.re.lastIndex = pos;
                    endMatch = end.re.exec(text);
                }

                let best = null, bestRule = null;
                for (const rule of rules) {
                    const re = rule.re ?? rule.begin;
                    re.lastIndex = pos;
                    const m = re.exec(text);
                    if (m && (best === null || m.index < best.index)) {
                        best = m;
                        bestRule = rule;
                    }
                }

                // the end pattern wins ties, like in TextMate
                if (endMatch && (best === null || endMatch.index <= best.index)) {
                    emit(text.slice(pos, endMatch.index), scope);
                    emit(endMatch[0], end.name ?? scope);
                    return endMatch.index + endMatch[0].length;
                }
                if (best === null || best[0].length === 0) {
                    break;
                }

                emit(text.slice(pos, best.index), scope);
                if (bestRule.re) {
                    emit(best[0], bestRule.name ?? scope);
                    pos = best.index + best[0].length;
                } else {
                    const inner = bestRule.name ?? scope;
                    emit(best[0], bestRule.beginName ?? inner);
                    pos = run(best.index + best[0].length, bestRule.patterns, {re: bestRule.end, name: bestRule.endName}, inner);
                }
            }
            emit(text.slice(pos), scope);
            return text.length;
        };

        run(0, this.patterns, null, undefined);
        return tokens;
    }
}

const BUILTIN_TYPES = new Set(["int", "uint", "bool", "positive", "negative", "nonzero", "any", "str"]);
const DOMAIN_KEYWORDS = new Set(["sync", "async", "clock"]);

function tokenClass(token, next) {
    const scope = token.scope ?? "";
    if (scope.startsWith("comment")) return "tk-com";
    if (scope.startsWith("string")) return "tk-str";
    if (scope.startsWith("constant.numeric")) return "tk-num";
    if (scope.startsWith("constant.language")) return "tk-lit";
    if (scope.startsWith("keyword.operator")) return "tk-op";
    if (scope.startsWith("storage") || scope.startsWith("keyword")) {
        return DOMAIN_KEYWORDS.has(token.text) ? "tk-dom" : "tk-kw";
    }
    if (scope.startsWith("identifier")) {
        if (BUILTIN_TYPES.has(token.text) || /^[A-Z]/.test(token.text)) return "tk-type";
        if (next && /^\s*\(/.test(next.text)) return "tk-fn";
    }
    return null;
}

function highlight(grammar, code) {
    const tokens = grammar.tokenize(code);
    return tokens.map((token, i) => {
        const cls = tokenClass(token, tokens[i + 1]);
        const text = escapeHtml(token.text);
        return cls ? `<span class="${cls}">${text}</span>` : text;
    }).join("");
}

// Convert the compiler's ANSI colored diagnostics into spans, so the colors can follow the page theme.
const ANSI_COLORS = {31: "red", 32: "green", 33: "yellow", 34: "blue", 36: "cyan", 91: "red", 92: "green", 93: "yellow", 94: "blue", 96: "cyan"};

function ansiToHtml(text) {
    let html = "";
    let bold = false, color = null;
    for (const part of text.split(/(\x1b\[[0-9;]*m)/)) {
        const m = /^\x1b\[([0-9;]*)m$/.exec(part);
        if (m) {
            for (const code of m[1].split(";").map(Number)) {
                if (code === 0) {
                    bold = false;
                    color = null;
                } else if (code === 1) {
                    bold = true;
                } else if (ANSI_COLORS[code]) {
                    color = ANSI_COLORS[code];
                }
            }
            continue;
        }
        if (part.length === 0) continue;
        const classes = [bold ? "ansi-bold" : null, color ? `ansi-${color}` : null].filter(c => c);
        html += classes.length ? `<span class="${classes.join(" ")}">${escapeHtml(part)}</span>` : escapeHtml(part);
    }
    return html;
}

function readExample(name) {
    const file = path.join(EXAMPLES_DIR, name);
    if (!fs.existsSync(file)) throw new Error(`example ${name} not found`);
    const content = fs.readFileSync(file, "utf8");
    const markerIndex = content.indexOf(HIDDEN_MARKER);
    const shown = markerIndex === -1 ? content : content.slice(0, markerIndex);
    const full = markerIndex === -1 ? content : content.slice(0, markerIndex) + content.slice(markerIndex + HIDDEN_MARKER.length).replace(/^\n/, "");
    return {shown: shown.trimEnd() + "\n", full: full.trimEnd() + "\n"};
}

// Same encoding as the share links of the web demo.
function demoHref(source) {
    const encoded = zlib.deflateSync(Buffer.from(source, "utf8")).toString("base64");
    return `demo/?source=${encodeURIComponent("cb64-" + encoded)}`;
}

function buildPage(grammar, template) {
    return template
        .replace(/<pre data-example="([^"]+)"><\/pre>/g, (_, name) => {
            const code = highlight(grammar, readExample(name).shown);
            return `<pre class="code"><code>${code}</code></pre>`;
        })
        .replace(/<pre data-diagnostic="([^"]+)"><\/pre>/g, (_, name) => {
            const errFile = path.join(EXAMPLES_DIR, name.replace(/\.kh$/, ".err"));
            if (!fs.existsSync(errFile)) throw new Error(`diagnostic for ${name} not found`);
            const text = fs.readFileSync(errFile, "utf8").trimEnd();
            return `<pre class="diagnostic"><code>${ansiToHtml(text)}</code></pre>`;
        })
        .replace(/data-demo="([^"]+)"/g, (_, name) => `href="${escapeHtml(demoHref(readExample(name).full))}"`);
}

function copyDir(from, to) {
    fs.mkdirSync(to, {recursive: true});
    for (const entry of fs.readdirSync(from, {withFileTypes: true})) {
        const src = path.join(from, entry.name);
        const dst = path.join(to, entry.name);
        if (entry.isDirectory()) {
            copyDir(src, dst);
        } else if (entry.name !== "index.html") {
            fs.copyFileSync(src, dst);
        }
    }
}

function main() {
    const {out} = parseArgs(process.argv.slice(2));
    const grammar = new Grammar(JSON.parse(fs.readFileSync(GRAMMAR_PATH, "utf8")));
    const template = fs.readFileSync(path.join(SRC_DIR, "index.html"), "utf8");
    const page = buildPage(grammar, template);

    copyDir(SRC_DIR, out);
    fs.writeFileSync(path.join(out, "index.html"), page);
    console.log(`Built website into ${out}`);
}

main();
