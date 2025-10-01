from pathlib import Path

from hwl_sandbox.common.expression import compile_expression


def test_add_pos(tmpdir: Path):
    e = compile_expression("int(0..16)", "int(0..32)", "int(0..48)", "a + b", tmpdir)

    e.eval_assert(0, 0, 0)
    e.eval_assert(0, 1, 1)
    e.eval_assert(15, 31, 46)
