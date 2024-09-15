/* --------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 * ------------------------------------------------------------------------------------------ */

import {ExtensionContext} from 'vscode';

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
