from pathlib import Path

from hwl_sandbox.common.compare import compare_compile
from hwl_sandbox.common.util import compile_custom


def test_conditional_write():
    src = """
    module foo ports(x: in async int(0..8)) {
        comb {
            var a = false;
            if (x < 4) {
                a = true;
            }
        }
    }
    """
    compile_custom(src).resolve("top.foo")


def test_merge_tuple(tmp_dir: Path):
    src = """
    val result;
    if (a0) {
        result = (false, 1);
    } else {
        result = (true, 0);
    }
    return result;
    """
    e = compare_compile(["bool"], "Tuple(bool, uint(1))", src, tmp_dir)
    e.eval_assert([True], (False, 1))
    e.eval_assert([False], (True, 0))


def test_merge_struct(tmp_dir: Path):
    src = """
    struct S { x: bool } 
    val result;
    if (a0) {
        result = S.new(x=false);
    } else {
        result = S.new(x=true);
    }
    return result.x;
    """
    e = compare_compile(["bool"], "bool", src, tmp_dir)
    e.eval_assert([True], False)
    e.eval_assert([False], True)
