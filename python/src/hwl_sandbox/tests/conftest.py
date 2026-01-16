import os
import subprocess
from pathlib import Path

import pytest
from _pytest._py.path import LocalPath

if "PYTEST_XDIST_WORKER" not in os.environ:
    # Tell verilator to use ccache if installed
    try:
        subprocess.check_output(["ccache", "--version"])
    except FileNotFoundError:
        pass
    else:
        os.environ["OBJCACHE"] = "ccache"


@pytest.fixture
def tmp_dir(tmpdir: LocalPath) -> Path:
    return Path(tmpdir)
