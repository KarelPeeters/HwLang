TODO visitor (shudders) design sketch:
the following LSP features all need some kind of AST visiting:

* go to definition: walk down, declaring things in scopes, skip things outside of certain span, return single potential
  result
* find usages: walk down, declaring things in scopes, don't skip anything, collect multiple results
* expand selection: walk down, building nested tree of spans, skip things outside of certain span, return span stack (
  effectively a single result)
* folding ranges: walk down, building
* find the list of declared top-level IDs in a file? we don't really need the full visitor for that
* autocomplete: find the full set of identifiers that is in scope at a certain position

* implement the opposite direction, "find usages"
* follow imports instead of jumping to them
* we can do better: in `if(_) { a } else { a }` a is not really conditional any more
* generalize this "visitor", we also want to collect all usages, find the next selection span, find folding ranges, ...
* maybe this should be moved to the LSP, the compiler itself really doesn't need this
* use the real Scope for the file root, to reduce duplication and get a guaranteed match
