import subprocess
from pathlib import Path

from hwl_sandbox.common.util_no_hwl import enable_rust_backtraces

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
