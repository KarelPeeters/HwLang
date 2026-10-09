from pathlib import Path

from hwl import Source


def test_webdemo_compiles():
    s = Source()

    path = Path(__file__).parent / "../../../../../design/top_webdemo.kh"
    s.add_file_content(["top"], "top.kh", path.read_text())

    c = s.compile()
    _ = c.resolve_module("top.top")
