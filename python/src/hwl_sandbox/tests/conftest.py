import os
import tempfile
from pathlib import Path
from typing import Generator

import hwl
import pytest
from _pytest._py.path import LocalPath

from hwl_sandbox.common.util import BuildSim
from hwl_sandbox.common.util_no_hwl import enable_ccache_if_available

# Only do these once in the root process (when using pytest-xdist)
if "PYTEST_XDIST_WORKER" not in os.environ:
    # enable_rust_backtraces()
    enable_ccache_if_available()


@pytest.fixture
def tmp_dir(tmpdir: LocalPath) -> Path:
    return Path(tmpdir)


@pytest.fixture(params=("verilator", "simulator"))
def build_sim(request: pytest.FixtureRequest) -> Generator[BuildSim]:
    kind = request.param

    if kind == "verilator":
        # verilator needs a directory for compilation artifacts,
        #   create separate temporary sub-directory for each call
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            next_index = 0

            def f(m: hwl.Module):
                nonlocal next_index

                curr_index = next_index
                next_index += 1

                tmp_path_curr = tmp_path / str(curr_index)
                tmp_path_curr.mkdir()
                return m.as_verilated(build_dir=tmp_path_curr)

            yield f
    elif kind == "simulator":
        # the builtin simulator does not need any directory
        def f(m: hwl.Module):
            return m.as_simulator()

        yield f
    else:
        assert False
