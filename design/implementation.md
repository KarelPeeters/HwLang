Useful crates:
* [salsa](http://salsa-rs.github.io/salsa/) compiler graph caching framework
* [codespan-reporting](https://docs.rs/codespan-reporting/0.9.0/codespan_reporting/) error reporting library
* [notify](https://docs.rs/notify/4.0.15/notify/) file watcher
* [egg](https://egraphs-good.github.io/) equivalence graphs: useful for checking equality of unbound expressions
* https://crates.io/crates/lsp-textdocument

Web demo:

* look at what other implementations do: REPL, rust playground, Kotlin playground, Zig, ...
* a good search term for this is "webassembly language server protocol"
* Maybe just use VSCode: https://code.visualstudio.com/blogs/2024/06/07/wasm-part2
* https://www.hiro.so/blog/write-clarity-smart-contracts-with-zero-installations-how-we-built-an-in-browser-language-server-using-wasm

Relevant projects:

* Rust-Analyzer for LSP architecture
  ideas https://github.com/rust-lang/rust-analyzer/blob/master/docs/dev/architecture.md

Websites and blog posts:

* Equivalence graphs: https://www.cole-k.com/2023/07/24/e-graphs-primer/
* Anything written or referenced by matklad (Rust-Analyzer and JB Rust plugin dev). In particular:
  * Language design
    * Module system
      * https://lobste.rs/s/u7y4lk/modules_matter_most_for_masses#c_i6a8n9
      * https://todo.sr.ht/~icefox/garnet/52#event-242650
      * https://matklad.github.io/2023/08/01/on-modularity-of-lexical-analysis.html
      * https://matklad.github.io/2021/11/27/notes-on-module-system.html
    * Type system
      * https://matklad.github.io/2023/08/09/types-and-zig.html
      * https://matklad.github.io/2019/07/25/unsafe-as-a-type-system.html
  * LSP
    * https://matklad.github.io/2023/03/08/an-engine-for-an-editor.html
    * https://matklad.github.io/2022/04/25/why-lsp.html
    * https://rust-analyzer.github.io/blog/2023/12/26/the-heart-of-a-language-server.html
  * Project management
    * https://matklad.github.io/2024/03/22/basic-things.html
    * https://matklad.github.io/2023/12/31/O(1)-build-file.html
    * https://matklad.github.io/2023/06/18/GitHub-merge-queue.html
    * https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html
  * Parsing
    * https://matklad.github.io/2023/05/21/resilient-ll-parsing-tutorial.html
  * Other
    * https://matklad.github.io/2023/02/21/why-SAT-is-hard.html
    * https://matklad.github.io/2020/03/22/fast-simple-rust-interner.html
