import os
import subprocess
from pathlib import Path

from hwl_sandbox.common.util_no_hwl import enable_rust_backtraces

# Only do these once in the root process (when using pytest-xdist)
if "PYTEST_XDIST_WORKER" not in os.environ:
    # Update the rust module
    # (be careful not to import hwl before this point)
    print("Installing/updating hwl_python rust module")
    subprocess.check_call(
        # pass some args to speed up maturin, but ideally it would be fully incremental
        ["maturin", "develop", "--compression-method", "stored", "--uv"],
        cwd=Path(__file__).parent / "../../../../rust/hwl_python/"
    )

    # Enable rust backtraces by default for easier debugging
    enable_rust_backtraces()
