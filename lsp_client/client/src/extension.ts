import {ExtensionContext} from 'vscode';
import {LanguageClient, LanguageClientOptions, ServerOptions} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    // TODO get this from a proper location
    const binary_path = "/home/karel/Documents/hwlang/rust/target/debug/hwl_lsp_server";

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

    // Create the language client and start the client.
    client = new LanguageClient(
        'languageServerExample',
        'Language Server Example',
        serverOptions,
        clientOptions
    );

    // Start the client. This will also launch the server
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
