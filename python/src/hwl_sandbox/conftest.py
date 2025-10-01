import os
import subprocess
from pathlib import Path

# Update the rust module
subprocess.check_call(
    # pass some args to speed up maturin, but ideally it would be fully incremental
    ["maturin", "develop", "--compression-method", "stored", "--uv"],
    cwd=Path(__file__).parent / "../../../rust/hwl_python/"
)

# Enable rust backtraces by default for easier debugging
if "RUST_BACKTRACE" not in os.environ:
    os.environ["RUST_BACKTRACE"] = "1"
