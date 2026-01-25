import os
from pathlib import Path

import pytest
from _pytest._py.path import LocalPath

from hwl_sandbox.common.util_no_hwl import compile_rust_module, enable_ccache_if_available

# Only do these once in the root process (when using pytest-xdist)
if "PYTEST_XDIST_WORKER" not in os.environ:
    if "PYTEST_SKIP_RUST_BUILD" not in os.environ:
        compile_rust_module()

    # enable_rust_backtraces()
    enable_ccache_if_available()


@pytest.fixture
def tmp_dir(tmpdir: LocalPath) -> Path:
    return Path(tmpdir)
