/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import {ExtensionContext, workspace} from 'vscode';

import {LanguageClient, LanguageClientOptions, ServerOptions} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
	// The server is implemented in node
	// const serverModule = context.asAbsolutePath(
	// 	path.join('server', 'out', 'server.js')
	// );

	// If the extension is launched in debug mode then the debug server options are used
	// Otherwise the run options are used

	// TODO get this from a proper location
    const binary_path = "C:/Documents/Programming/HDL/hwlang/rust/target/debug/hwl_lsp_server.exe";

	const serverOptions: ServerOptions = {
		// run: { module: serverModule, transport: TransportKind.ipc },
		// debug: {
		// 	module: serverModule,
		// 	transport: TransportKind.ipc,
		// }
		run: { command: binary_path },
		debug: { command: binary_path },
	};

	// Options to control the language client
	const clientOptions: LanguageClientOptions = {
		// Register the server for plain text documents
		// documentSelector: [{ scheme: 'file', language: 'plaintext' }],
		documentSelector: [
			{ scheme: 'file', language: 'plaintext' },
			{ pattern: "**/*.kh" },
		],
		synchronize: {
            // Notify the server about relevant file changes in the workspace
            // TODO move this to the server, it knows better
            // TODO double-check whether this also notifies for deletion, creation and movement
            //   the end goal is to have a consistent view of the file system in the LSP server
            // TODO how does the LSP server get all initial file contents? just its own traversal?
            // TODO add a custom LSP acion to force a full refresh
            fileEvents: workspace.createFileSystemWatcher('**/{*.kh,.kh_config.toml}')
		}
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
