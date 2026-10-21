#!/usr/bin/env bash
set -ue

cd "$(dirname "$0")"

# Build wasm binaries
(cd ../rust/khdl_wasm && wasm-pack build --dev --target web)

# Install dependencies
npm ci
# Build and run server
npm run start
