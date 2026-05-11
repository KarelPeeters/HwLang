# HWLang JetBrains LSP client

JetBrains IDE plugin for connecting `.kh` files to the existing `hwl_lsp_server` binary over stdio.

## Requirements

- A supported commercial JetBrains IDE with the public LSP API
- Java 21
- A built `hwl_lsp_server` executable, configured in the plugin settings

This plugin does **not** bundle the language server. It follows the same external-binary model as the existing VS Code client in this repository.

## Build

```bash
cd lsp_client_jetbrains
./gradlew buildPlugin
```

## Run in a development IDE

```bash
cd lsp_client_jetbrains
./gradlew runIde
```

Then open **Settings | Tools | HWLang LSP** and configure the path to the `hwl_lsp_server` executable, for example:

```bash
/absolute/path/to/hwlang2/rust/target/debug/hwl_lsp_server
```

The plugin restarts the server automatically when:

- the configured server path changes in settings
- the configured binary is replaced or modified on disk

