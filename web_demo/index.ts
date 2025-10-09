import {EditorState} from "@codemirror/state"
import {EditorView, highlightActiveLineGutter, keymap, lineNumbers} from "@codemirror/view"
import {defaultKeymap, history, historyKeymap, indentWithTab, insertNewlineKeepIndent} from "@codemirror/commands"
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
import {verilog as mode_verilog} from "@codemirror/legacy-modes/mode/verilog";
import {cpp as mode_cpp} from "@codemirror/legacy-modes/mode/clike";

import AnsiToHtmlClass from "ansi-to-html";
import Cookies from "js-cookie";
import pako from "pako";

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

const element_editor_input = document.getElementById("div-editor-input");
const element_editor_output_verilog = document.getElementById("div-editor-output-verilog");
const element_editor_output_cpp = document.getElementById("div-editor-output-cpp");
const element_editor_output_format = document.getElementById("div-editor-output-format");
const element_diags_compile = document.getElementById("div-diags-compile");
const element_diags_format = document.getElementById("div-diags-format");
const element_share_link = document.getElementById("button-share") as HTMLAnchorElement;
const element_clear_button = document.getElementById("button-clear");

const ansi_to_html = new AnsiToHtmlClass();

function escapeHtml(raw: string): string {
    return raw
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function diagnostics_ansi_to_html(ansi: string): string {
    if (ansi.length == 0) {
        return "";
    }

    let result = "";
    for (let line of ansi.split("\n")) {
        if (line.length == 0) {
            result += "<div>&ZeroWidthSpace;</div>";
        } else {
            result += "<div>" + ansi_to_html.toHtml(escapeHtml(line).replaceAll(" ", "&nbsp;")) + "</div>";
        }
    }
    return result;
}

const EMPTY_DOC = "// empty";
const COOKIE_SOURCE = "source";

function onDocumentChanged(source: string, editor_view_output_verilog: EditorView, editor_view_output_cpp: EditorView, editor_view_output_format: EditorView, format_visible: boolean) {
    // store code in cookie
    Cookies.set(COOKIE_SOURCE, source);

    // store code in share link, as the parameter "source", compressed and base64 encoded
    // https://developer.mozilla.org/en-US/docs/Glossary/Base64#the_unicode_problem
    const url = new URL(window.location.href);
    url.searchParams.set("source", "cb64-" + btoa(String.fromCharCode.apply(null, pako.deflate(source))));
    element_share_link.href = url.toString()

    // run the compiler
    let compile_diags_ansi, lowered_verilog, lowered_cpp, format_diags_ansi, format_debug_str;
    try {
        const result = hwl_wasm.run_all(source, format_visible);
        compile_diags_ansi = result.compile_diags_ansi;
        lowered_verilog = result.lowered_verilog;
        lowered_cpp = result.lowered_cpp;
        format_diags_ansi = result.format_diags_ansi;
        format_debug_str = result.format_debug_str;

        if (result.compile_prints.length > 0) {
            let combined_prints = "// prints:\n";
            for (const print of result.compile_prints) {
                combined_prints += "//  " + print + "\n";
            }
            combined_prints += "\n";
            lowered_verilog = combined_prints + lowered_verilog;
        }
    } catch (e) {
        compile_diags_ansi = "compiler panicked\nsee console for the error message and stack trace";
        lowered_verilog = "";
        lowered_cpp = "";
        format_diags_ansi = compile_diags_ansi;
        format_debug_str = "";
    }

    // display diagnostics as html
    element_diags_compile.innerHTML = diagnostics_ansi_to_html(compile_diags_ansi);
    element_diags_format.innerHTML = diagnostics_ansi_to_html(format_diags_ansi);

    // replace output contents with newly generated source,
    // put at least some text to prevent confusion
    if (lowered_verilog.length == 0) {
        lowered_verilog = EMPTY_DOC;
    }
    if (lowered_cpp.length == 0) {
        lowered_cpp = EMPTY_DOC;
    }
    if (format_debug_str.length == 0) {
        format_debug_str = EMPTY_DOC;
    }

    editor_view_output_verilog.dispatch({
        changes: {
            from: 0,
            to: editor_view_output_verilog.state.doc.length,
            insert: lowered_verilog,
        }
    })
    editor_view_output_cpp.dispatch({
        changes: {
            from: 0,
            to: editor_view_output_cpp.state.doc.length,
            insert: lowered_cpp,
        }
    })
    editor_view_output_format.dispatch({
        changes: {
            from: 0,
            to: editor_view_output_format.state.doc.length,
            insert: format_debug_str,
        }
    })
}

let common_extensions = [
    keymap.of([{key: "Enter", run: insertNewlineKeepIndent}]),
    keymap.of([indentWithTab]),
    keymap.of(defaultKeymap),
    keymap.of(historyKeymap),
    history(),

    lineNumbers(),
    bracketMatching(),

    indentUnit.of(" ".repeat(4)),
    syntaxHighlighting(defaultHighlightStyle),
];

function formatCurrentCode() {
    const currentCode = editor_view_input.state.doc.toString();
    const formatted = hwl_wasm.format_source(currentCode);
    if (formatted !== undefined) {
        editor_view_input.dispatch({
            changes: {
                from: 0,
                to: editor_view_input.state.doc.length,
                insert: formatted,
            }
        });
    }
}

// TODO compare legacy mode to to https://www.npmjs.com/package/codemirror-lang-verilog
let editor_state_output_verilog = EditorState.create({
    doc: EMPTY_DOC,
    extensions: common_extensions.concat([
        EditorState.readOnly.of(true),
        StreamLanguage.define(mode_verilog)
    ]),
})
let editor_view_output_verilog = new EditorView({
    state: editor_state_output_verilog,
    parent: element_editor_output_verilog
})
let editor_state_output_cpp = EditorState.create({
    doc: EMPTY_DOC,
    extensions: common_extensions.concat([
        EditorState.readOnly.of(true),
        StreamLanguage.define(mode_cpp)
    ]),
})
let editor_view_output_cpp = new EditorView({
    state: editor_state_output_cpp,
    parent: element_editor_output_cpp
})
let editor_state_output_format = EditorState.create({
    doc: EMPTY_DOC,
    extensions: common_extensions.concat([
        EditorState.readOnly.of(true),
        new LanguageSupport(language)
    ]),
})
let editor_view_output_format = new EditorView({
    state: editor_state_output_format,
    parent: element_editor_output_format
})

// TODO get this out of the typing event loop, run this async or on a separate thread
let format_visible = false;

function force_update() {
    onDocumentChanged(editor_view_input.state.doc.toString(), editor_view_output_verilog, editor_view_output_cpp, editor_view_output_format, format_visible);
}

let updateListenerExtension = EditorView.updateListener.of((update) => {
    if (update.docChanged) {
        onDocumentChanged(editor_view_input.state.doc.toString(), editor_view_output_verilog, editor_view_output_cpp, editor_view_output_format, format_visible);
    }
})

const wasm_initial_source = hwl_wasm.initial_source();
let initial_source = wasm_initial_source;
{
    let cookie_doc = Cookies.get(COOKIE_SOURCE);
    // get source from cookie
    if (cookie_doc != undefined) {
        initial_source = cookie_doc;
    }

    // get source from current URL
    const url = new URL(window.location.href);
    const source = url.searchParams.get("source");
    if (source != null && source.startsWith("cb64-")) {
        initial_source = pako.inflate(Uint8Array.from(atob(source.slice(5)), c => c.charCodeAt(0)), {to: "string"});
    }
}

let editor_state_input = EditorState.create({
    doc: initial_source,
    extensions: common_extensions.concat([
        highlightActiveLineGutter(),
        new LanguageSupport(language),
        updateListenerExtension,
        keymap.of([{
            key: "Ctrl-Shift-i", run: () => {
                formatCurrentCode();
                return true;
            }
        }]),
    ])
})
let editor_view_input = new EditorView({
    state: editor_state_input,
    parent: element_editor_input,
})

// add clear event handler
element_clear_button.addEventListener("click", () => {
    editor_view_input.dispatch({
        changes: {
            from: 0,
            to: editor_view_input.state.doc.length,
            insert: wasm_initial_source,
        }
    })
});

// add tab change event handlers
function show_output_tab(tab: string) {
    function set_element_display(id: string, filter: string[]) {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = filter.includes(tab) ? 'block' : 'none';
        }
    }

    set_element_display("div-diags-compile", ["verilog", "cpp"])
    set_element_display("div-editor-output-verilog", ["verilog"])
    set_element_display("div-editor-output-cpp", ["cpp"])

    set_element_display("div-diags-format", ["format"])
    set_element_display("div-editor-output-format", ["format"])

    // update format tab state and possible compute format state if it was not visible before
    let format_was_visible = format_visible;
    format_visible = (tab == "format");
    if (format_visible && !format_was_visible) {
        force_update();
    }
}

const svg_verilog = document.getElementById("svg-verilog");
const svg_cpp = document.getElementById("svg-cpp");
const svg_format = document.getElementById("svg-format");
svg_verilog.addEventListener("click", () => show_output_tab("verilog"));
svg_cpp.addEventListener("click", () => show_output_tab("cpp"));
svg_format.addEventListener("click", () => show_output_tab("format"));

// initial update
force_update();
