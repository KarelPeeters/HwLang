import {ExtensionContext} from 'vscode';
import {LanguageClient, LanguageClientOptions, ServerOptions} from 'vscode-languageclient/node';
import * as fs from "node:fs";
import * as path from "node:path";

// TODO get this from a proper location
const binary_path = "/home/karel/Documents/hwlang/rust/target/debug/hwl_lsp_server";

let client: LanguageClient;

// Restart the client when the server binary changes, to emulate hot-reloading.
let watcher: fs.FSWatcher;
function register_watcher(context: ExtensionContext) {
    // Watch the parent folder instead of the binary itself so delete+move is detected too.
    watcher = fs.watch(path.dirname(binary_path), (eventType, _filename) => {
        if (eventType === 'change' && client) {
            client.stop().then(() => client.start());
        }
    });

    context.subscriptions.push({
        dispose: () => {
            watcher.close();
        }
    });
}

export function activate(context: ExtensionContext) {
    register_watcher(context);

    const serverOptions: ServerOptions = {
        run: {command: binary_path},
        debug: {command: binary_path},
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            {scheme: 'file', language: 'hwlang'},
            // { scheme: 'file', pattern: "**/*.kh" },
        ],
        // file watching is registered by the server, no need to repeat it in the client plugin
        synchronize: {}
    };

    client = new LanguageClient(
        'hwl-lsp',
        'HwLang LSP',
        serverOptions,
        clientOptions
    );
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
