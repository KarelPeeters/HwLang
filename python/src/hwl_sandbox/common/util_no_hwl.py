import os
import subprocess
from pathlib import Path

import sys


def compile_rust_module():
    """Update the rust module, be careful not to import hwl before this point"""
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


def enable_rust_backtraces():
    var = "RUST_BACKTRACE"
    if var not in os.environ:
        os.environ[var] = "1"
