from pathlib import Path

from hwl_sandbox.common.compare import compare_compile


def test_bool_to_bits(tmp_dir: Path):
    c = compare_compile(["bool"], "[1]bool", "return bool.to_bits(a0);", tmp_dir)
    c.eval_assert([False], [False])
    c.eval_assert([True], [True])
