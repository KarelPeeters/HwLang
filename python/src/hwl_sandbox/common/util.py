import hwl


def compile_custom(top: str) -> hwl.Compile:
    # TODO compile/elaborate everything by default
    source = hwl.Source()
    source.add_file_content(["top"], "python.kh", top)
    return source.compile()
