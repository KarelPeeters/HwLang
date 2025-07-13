from pathlib import Path

import hwl


def compile_custom(top: str) -> hwl.Compile:
    # TODO compile/elaborate everything by default
    builder = hwl.SourceBuilder()
    builder.add_tree(["std"], str(Path(__file__).parent / "../../design/project/std"))
    builder.add_file(["top"], "python.top", top)
    return builder.finish().compile()
