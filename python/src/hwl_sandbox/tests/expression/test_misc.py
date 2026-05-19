import pytest

import hwl
from hwl_sandbox.common.util import compile_custom


def test_dummy():
    src = "const c = _;"
    with pytest.raises(hwl.DiagnosticException, match="dummy expression not allowed in this context"):
        _ = compile_custom(src).resolve("top.c")
