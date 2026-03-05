import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_parse_clear_error():
    # this code is missing a closing parenthesis after uint, so the error should be close to that
    # (at some point we returned the wrong parser error in cases where error recovery happened)
    src = """
    fn foo(x: uint() -> bool {
        return x > 0;
    }
    
    fn bar() {
        val v = 8;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="unexpected token `->`"):
        compile_custom(src)
