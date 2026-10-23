# Website

The landing page of https://khdl-lang.org/. CI builds it into the root of the site, with the web demo at `/demo/`.

* `src/`: the page template, styles and the animated figures (plain JavaScript modules, no dependencies).
* `examples/`: the KHDL code shown on the page.
* `build.mjs`: fills the code examples, compiler diagnostics and demo links into the template.

Build and preview locally (Node 20 or newer), with the web demo at `/demo/` like on the real site:

```bash
node website/build.mjs
(cd rust/khdl_wasm && npx wasm-pack build --target web)  # needs the wasm32-unknown-unknown Rust target
(cd web_demo && npm ci && npm run build)
cp -r web_demo/dist/. website/dist/demo/
python3 -m http.server -d website/dist 8000
```

After changing the landing page, only the first command needs to run again. After changing the demo, rebuild it and
copy it again.

## Code examples

The page refers to examples by file name: `<pre data-example="counter.kh"></pre>` shows the code,
`<pre data-diagnostic="counter_overflow.kh"></pre>` shows the compiler output stored in `counter_overflow.err`, and
`data-demo="counter.kh"` on a link opens the example in the web demo. Code after the line
`// ---- website: hidden below this line ----` is only included in the demo link, for example wrapper modules for
generic modules.

`python/src/khdl_sandbox/tests/misc/test_website_examples.py` checks that every example is formatted, that its
non-generic public modules elaborate and export to Verilog, and that examples with an `.err` file fail with exactly
that diagnostic. After an intended change to the compiler output, regenerate the `.err` files with:

```bash
cd python
KHDL_UPDATE_WEBSITE_EXAMPLES=1 uv run pytest src/khdl_sandbox/tests/misc/test_website_examples.py
```
