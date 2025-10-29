TODO visitor (shudders) design sketch:
the following LSP features all need some kind of AST visiting:

[//]: # (TODO make find defs and find usages cross-file)

* find the list of declared top-level IDs in a file? we don't really need the full visitor for that
* autocomplete: find the full set of identifiers that is in scope at a certain position

* follow imports instead of jumping to them
* get "go to declaration" working for function calls, module ports, ...
* we can do better: in `if(_) { a } else { a }` a is not really conditional any more
