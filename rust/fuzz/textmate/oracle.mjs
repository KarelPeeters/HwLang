// Tokenizes source code with vscode-textmate, the TextMate engine used by VS Code.
//
// Protocol, one JSON value per line:
//   * stdin, first line: the grammar, as a JSON string containing the grammar JSON
//   * stdin, every next line: a source file, as a JSON string
//   * stdout, one line per source file: an array of `[start_byte, end_byte, scopes]` tokens,
//     with byte offsets into the full source file and `scopes` ordered from outer to inner.
//
// Lines are split on `\n` only, the caller should avoid other line terminators.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { createInterface } from "node:readline";
import oniguruma from "vscode-oniguruma";
import vsctm from "vscode-textmate";

const require = createRequire(import.meta.url);
const wasm = readFileSync(require.resolve("vscode-oniguruma/release/onig.wasm"));
await oniguruma.loadWASM(wasm.buffer);

const onigLib = Promise.resolve({
    createOnigScanner: (patterns) => new oniguruma.OnigScanner(patterns),
    createOnigString: (s) => new oniguruma.OnigString(s),
});

const lines = createInterface({ input: process.stdin, terminal: false });
let grammar = null;

for await (const line of lines) {
    const text = JSON.parse(line);

    if (grammar === null) {
        const raw = vsctm.parseRawGrammar(text, "grammar.json");
        const registry = new vsctm.Registry({
            onigLib,
            loadGrammar: async (scopeName) => (scopeName === raw.scopeName ? raw : null),
        });
        grammar = await registry.loadGrammar(raw.scopeName);
        continue;
    }

    process.stdout.write(JSON.stringify(tokenize(grammar, text)) + "\n");
}

function tokenize(grammar, source) {
    const result = [];
    let ruleStack = vsctm.INITIAL;
    let lineStartByte = 0;

    for (const line of source.split("\n")) {
        const lineResult = grammar.tokenizeLine(line, ruleStack);
        ruleStack = lineResult.ruleStack;

        // convert utf-16 offsets within the line to utf-8 offsets within the source
        const toByte = (index) => lineStartByte + Buffer.byteLength(line.slice(0, Math.min(index, line.length)));
        for (const token of lineResult.tokens) {
            const start = toByte(token.startIndex);
            const end = toByte(token.endIndex);
            if (start < end) {
                result.push([start, end, token.scopes]);
            }
        }

        lineStartByte += Buffer.byteLength(line) + 1;
    }

    return result;
}
