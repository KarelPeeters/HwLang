# Build wasm binaries
(cd ../rust/khdl_wasm && wasm-pack build --dev)

# Start npm server (this includes building NPM stuff automatically)
npm run start