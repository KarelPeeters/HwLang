from pathlib import Path

from hwl_sandbox.common.compare import compare_body
from hwl_sandbox.common.util import compile_custom, diag_error


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
    with diag_error("unexpected token", has_message="unexpected token `->`"):
        compile_custom(src)


def test_parse_ceil_div_comment(tmp_dir: Path):
    # `+/` must not be tokenized when followed by the start of a comment
    body = """
    val x = a0 +// comment
        a1;
    return x +/* comment */ a1;
    """
    e = compare_body(["int(0..4)", "int(0..4)"], "int(0..12)", body, tmp_dir)
    e.eval_assert([1, 2], 5)
