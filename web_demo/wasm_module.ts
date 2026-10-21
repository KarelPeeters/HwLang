import init from "khdl_wasm";

// Compile the wasm module once, so compile workers can instantiate it without fetching and compiling it again.
export const wasm_module = await WebAssembly.compileStreaming(fetch(new URL("khdl_wasm/khdl_wasm_bg.wasm", import.meta.url)));
await init({module_or_path: wasm_module});
