from pathlib import Path

import hwl


def compile_custom(top: str) -> hwl.Compile:
    # TODO compile/elaborate everything by default
    source = hwl.Source()
    source.add_tree(["std"], str(Path(__file__).parent / "../../design/project/std"))
    source.add_file_content(["top"], "python.top", top)
    return source.compile()
