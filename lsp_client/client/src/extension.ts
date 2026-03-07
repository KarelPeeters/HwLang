import {ExtensionContext, workspace} from 'vscode';
import {LanguageClient, LanguageClientOptions, ServerOptions} from 'vscode-languageclient/node';
import * as fs from "node:fs";
import * as path from "node:path";

let client: LanguageClient | undefined;
let watcher: fs.FSWatcher | undefined;

function getServerPath(): string | undefined {
    return workspace.getConfiguration('hwlang.server').get<string>('path') || undefined;
}

function disposeWatcher() {
    if (watcher) {
        watcher.close();
        watcher = undefined;
    }
}

async function stopClient() {
    if (client) {
        await client.stop();
        client = undefined;
    }
}

async function startClient(context: ExtensionContext) {
    // stop the existing client if any
    disposeWatcher();
    await stopClient();

    // figure out the server path
    const serverPath = getServerPath();
    if (!serverPath) {
        // log a warning here (but don't show a popup)
        console.warn('HwLang LSP: No server path configured, LSP client not started');
        return;
    }

    // register watcher to handle the server binary updating
    //   watch the parent folder instead of the binary itself so delete+move is detected too.
    watcher = fs.watch(path.dirname(serverPath), (eventType, _filename) => {
        if (eventType === 'change' && client) {
            client.stop().then(() => client?.start());
        }
    });
    context.subscriptions.push({dispose: disposeWatcher});

    // start the client
    const serverOptions: ServerOptions = {
        run: {command: serverPath},
        debug: {command: serverPath},
    };
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            {scheme: 'file', language: 'hwlang'},
        ],
    };
    client = new LanguageClient(
        'hwl-lsp',
        'HwLang LSP',
        serverOptions,
        clientOptions
    );
    client.start();
}

// noinspection JSUnusedGlobalSymbols
export function activate(context: ExtensionContext) {
    // restart the client when the configured path changes
    context.subscriptions.push(
        workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('hwlang.server.path')) {
                startClient(context);
            }
        })
    );

    // start the initial client
    startClient(context);
}

// noinspection JSUnusedGlobalSymbols
export function deactivate(): Thenable<void> | undefined {
    disposeWatcher();
    if (!client) {
        return undefined;
    }
    return client.stop();
}
