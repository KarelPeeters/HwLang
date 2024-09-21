import {EditorState} from "@codemirror/state"
import {EditorView, highlightActiveLineGutter, keymap, lineNumbers} from "@codemirror/view"
import {defaultKeymap, history, indentWithTab} from "@codemirror/commands"
import {
    bracketMatching,
    defaultHighlightStyle,
    indentUnit,
    Language,
    LanguageSupport,
    syntaxHighlighting
} from "@codemirror/language"
import {Input, NodeSet, NodeType, Parser, PartialParse, Tree} from "@lezer/common"
import {styleTags, tags} from "@lezer/highlight"

import {codemirror_node_types, codemirror_tokenize_to_tree} from "hwl_wasm";

function build_node_types() {
    const node_types_string = codemirror_node_types();

    // build node types
    const child_node_types = node_types_string.map((name, index) => {
        return NodeType.define({id: index, name: name, top: false})
    })
    const top_node_type = NodeType.define({id: child_node_types.length, name: "top", top: true})
    let all_node_types = child_node_types.concat([top_node_type]);

    // create set, including styles
    const style_tags_object: any = {};
    const tags_any: any = tags;
    for (const name of codemirror_node_types()) {
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
            buffer: Array.from(codemirror_tokenize_to_tree(input_str)),
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

let startState = EditorState.create({
    doc: "Hello World changed",
    extensions: [
        keymap.of(defaultKeymap),
        keymap.of([indentWithTab]),
        history(),

        lineNumbers(),
        highlightActiveLineGutter(),
        bracketMatching(),

        indentUnit.of(" ".repeat(4)),
        syntaxHighlighting(defaultHighlightStyle),
        new LanguageSupport(language),
    ]
})

new EditorView({
    state: startState,
    parent: document.body
})

