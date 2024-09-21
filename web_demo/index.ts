import {EditorState} from "@codemirror/state"
import {EditorView, highlightActiveLineGutter, keymap, lineNumbers} from "@codemirror/view"
import {defaultKeymap, history, indentWithTab} from "@codemirror/commands"
import {
    bracketMatching,
    defaultHighlightStyle,
    defineLanguageFacet,
    indentUnit,
    Language,
    languageDataProp,
    LanguageSupport,
    StreamLanguage,
    syntaxHighlighting
} from "@codemirror/language"
import {Input, NodeSet, NodeType, Parser, PartialParse, Tree} from "@lezer/common"
import {styleTags, tags} from "@lezer/highlight"
import * as hwl_wasm from "hwl_wasm";
import {verilog} from "@codemirror/legacy-modes/mode/verilog";

import AnsiToHtmlClass from "ansi-to-html";

const topNode = NodeType.define({
    id: 0,
    name: "topNode",
    top: true,
});

function build_node_types() {
    const node_types_string = hwl_wasm.codemirror_node_types();

    // build node types
    const child_node_types = node_types_string.map((name, index) => {
        return NodeType.define({id: index, name: name, top: false})
    })
    const top_node_type = NodeType.define({
        id: child_node_types.length, name: "top", top: true, props: [[
            languageDataProp,
            defineLanguageFacet({
                commentTokens: {line: "//", block: {start: "/*", end: "*/"}}
            }),
        ]]
    })
    let all_node_types = child_node_types.concat([top_node_type]);

    // create set, including styles
    const style_tags_object: any = {};
    const tags_any: any = tags;
    for (const name of hwl_wasm.codemirror_node_types()) {
        style_tags_object[name] = tags_any[name];
    }
    const node_set = new NodeSet(all_node_types).extend(styleTags(style_tags_object));

    return {node_set, top_node_type};
}

const {node_set: NODE_SET, top_node_type: TOP_NODE_TYPE} = build_node_types();

// CodeMirror supports incremental parsing.
// We only have a batch parser implemented in Rust though, so we also do a full parse here.
//
// Implementation based on
// https://thetrevorharmon.com/blog/connecting-antlr-to-code-mirror-6-connecting-a-language-server/
class HwlParser extends Parser {
    createParse(input: Input): PartialParse {
        return this.startParse(input)
    }

    startParse(input: Input | string): PartialParse {
        const input_str = typeof input === "string" ? input : input.read(0, input.length);
        let input_length = input_str.length;

        const tree = Tree.build({
            buffer: Array.from(hwl_wasm.codemirror_tokenize_to_tree(input_str)),
            nodeSet: NODE_SET,
            topID: TOP_NODE_TYPE.id,
        });

        return {
            stoppedAt: input_length,
            parsedPos: input_length,
            stopAt: (_) => {
            },
            advance: () => tree,
        };
    }
}

let language = new Language(null, new HwlParser(), [], "HWLang");

const ansi_to_html = new AnsiToHtmlClass();
const diagnostics_element = document.getElementById("split-mid");

function escapeHtml(raw: string): string {
    return raw
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function diagnostics_ansi_to_html(ansi: string): string {
    // TODO escape inner html
    let result = "";
    for (let line of ansi.split("\n")) {
        result += "<div>" + ansi_to_html.toHtml(escapeHtml(line).replace(/ /g, "&nbsp;")) + "</div>";
    }
    return result;
}

function onDocumentChanged(source: string, editor_view_verilog: EditorView) {
    // run the compiler
    let {diagnostics_ansi, lowered_verilog} = hwl_wasm.compile_and_lower(source);

    // display diagnostics as html
    diagnostics_element.innerHTML = diagnostics_ansi_to_html(diagnostics_ansi);

    // replace output content with newly generated verilog,
    // put at least some text to prevent confusion
    if (lowered_verilog.length == 0) {
        lowered_verilog = "// empty";
    }
    editor_view_verilog.dispatch({
        changes: {
            from: 0,
            to: editor_view_verilog.state.doc.length,
            insert: lowered_verilog,
        }
    })
}

let common_extensions = [
    keymap.of(defaultKeymap),
    keymap.of([indentWithTab]),
    history(),

    lineNumbers(),
    bracketMatching(),

    indentUnit.of(" ".repeat(4)),
    syntaxHighlighting(defaultHighlightStyle),
];

// TODO compare legacy mode to to https://www.npmjs.com/package/codemirror-lang-verilog
let editor_state_verilog = EditorState.create({
    doc: "module foo; endmodule;",
    extensions: common_extensions.concat([
        EditorState.readOnly.of(true),
        StreamLanguage.define(verilog)
    ]),
})
let editor_view_verilog = new EditorView({
    state: editor_state_verilog,
    parent: document.getElementById("split-right")
})

// TODO get this out of the typing event loop, run this async or on a separate thread
let updateListenerExtension = EditorView.updateListener.of((update) => {
    if (update.docChanged) {
        onDocumentChanged(update.state.doc.toString(), editor_view_verilog);
    }
})

let editor_state_hdl = EditorState.create({
    doc: hwl_wasm.initial_source(),
    extensions: common_extensions.concat([
        highlightActiveLineGutter(),
        new LanguageSupport(language),
        updateListenerExtension,
    ])
})
let editor_view_hdl = new EditorView({
    state: editor_state_hdl,
    parent: document.getElementById("split-left"),
})

// initial update
onDocumentChanged(editor_view_hdl.state.doc.toString(), editor_view_verilog)
