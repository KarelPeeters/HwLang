import os
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest._py.path import LocalPath

from hwl_sandbox.common.util_no_hwl import enable_rust_backtraces

# Only do these once in the root process (when using pytest-xdist)
if "PYTEST_SKIP_RUST_BUILD" not in os.environ and "PYTEST_XDIST_WORKER" not in os.environ:
    # Update the rust module
    #   (be careful not to import hwl before this point)
    assert "hwl" not in sys.modules, "The hwl module was imported before it has been recompiled"

    try:
        subprocess.check_output(
            # pass some args to speed up maturin, but ideally it would be fully incremental
            ["maturin", "develop", "--compression-method", "stored", "--uv"],
            cwd=Path(__file__).parent / "../../../../rust/hwl_python/",
            stderr=subprocess.STDOUT,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise

    # Enable rust backtraces by default for easier debugging
    enable_rust_backtraces()

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
