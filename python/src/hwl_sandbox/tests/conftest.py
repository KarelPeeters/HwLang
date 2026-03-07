import os
from pathlib import Path

import pytest
from _pytest._py.path import LocalPath

from hwl_sandbox.common.util_no_hwl import enable_ccache_if_available

# Only do these once in the root process (when using pytest-xdist)
if "PYTEST_XDIST_WORKER" not in os.environ:
    # enable_rust_backtraces()
    enable_ccache_if_available()


@pytest.fixture
def tmp_dir(tmpdir: LocalPath) -> Path:
    return Path(tmpdir)
