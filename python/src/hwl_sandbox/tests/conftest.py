import os
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest._py.path import LocalPath


def compile_rust_module():
    """Update the rust module, be careful not to import hwl before this point)"""

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


def enable_ccache_if_available():
    """Tell verilator to use ccache if installed."""
    try:
        subprocess.check_output(["ccache", "--version"])
    except FileNotFoundError:
        pass
    else:
        os.environ["OBJCACHE"] = "ccache"


# Only do these once in the root process (when using pytest-xdist)
if "PYTEST_XDIST_WORKER" not in os.environ:
    if "PYTEST_SKIP_RUST_BUILD" not in os.environ:
        compile_rust_module()

    # Enable rust backtraces by default for easier debugging
    # enable_rust_backtraces()

    enable_ccache_if_available()


@pytest.fixture
def tmp_dir(tmpdir: LocalPath) -> Path:
    return Path(tmpdir)
