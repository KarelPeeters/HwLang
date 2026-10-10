import re
from pathlib import Path

from hwl_sandbox.common.util import compile_custom


def test_webdemo():
    # get webdemo source
    path = Path(__file__).parent / "../../../../../design/top_webdemo.kh"
    source = path.read_text()

    # check that it compiles and elaborates
    c = compile_custom(source)
    _ = c.resolve_module("top.top")


def test_readme():
    # get readme example code block
    path = Path(__file__).parent / "../../../../../README.md"
    blocks = re.findall(r"^```[^\n]*\n(.*?)^```", path.read_text(), flags=re.MULTILINE | re.DOTALL)
    assert len(blocks) == 1
    source = blocks[0]

    # check that it compiles and elaborates
    c = compile_custom(source)
    _ = c.resolve_module("top.top")
