// Thin wrapper around the real KHDL compiler, compiled to WebAssembly.

import init, * as khdl from "../vendor/khdl_wasm/khdl_wasm.js";

let types = null;
const compileCache = new Map();

export async function initKhdl() {
  await init({ module_or_path: new URL("../vendor/khdl_wasm/khdl_wasm_bg.wasm", import.meta.url) });
  types = khdl.codemirror_node_types();
}

// [{s, e, kind}] leaf tokens from the compiler's own tokenizer
export function khdlTokens(src) {
  const buf = khdl.codemirror_tokenize_to_tree(src);
  const out = [];
  for (let i = 0; i + 3 < buf.length; i += 4) {
    const kind = types[buf[i]];
    if (kind === undefined) continue;
    out.push({ s: buf[i + 1], e: buf[i + 2], kind });
  }
  out.sort((a, b) => a.s - b.s);
  return out;
}

// run the full compiler: diagnostics (ANSI) + lowered Verilog
export function compile(src) {
  let r = compileCache.get(src);
  if (r) return r;
  const t0 = performance.now();
  try {
    const res = khdl.run_all(src, false, false);
    r = { diags: res.compile_diags_ansi, verilog: res.lowered_verilog, ms: performance.now() - t0 };
    res.free();
  } catch (e) {
    r = { diags: "compiler panicked", verilog: "", ms: 0 };
  }
  compileCache.set(src, r);
  return r;
}

export function formatSource(src) {
  return khdl.format_source(src);
}
