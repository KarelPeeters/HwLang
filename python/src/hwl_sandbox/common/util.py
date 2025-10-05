import os
from pathlib import Path

import hwl


def compile_custom(top: str) -> hwl.Compile:
    # TODO compile/elaborate everything by default
    # TODO include the stdlib by default in source, or at least create a factory method for it
    source = hwl.Source()
    source.add_tree(["std"], str(Path(__file__).parent / "../../../../design/project/std"))
    source.add_file_content(["top"], "python.kh", top)
    return source.compile()


def enable_rust_backtraces():
    var = "RUST_BACKTRACE"
    if var not in os.environ:
        os.environ[var] = "1"
