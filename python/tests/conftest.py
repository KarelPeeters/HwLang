import os
import subprocess
from pathlib import Path

# Update the rust module
subprocess.check_call(["maturin", "develop"], cwd=Path(__file__).parent / "../../rust/hwl_python/")

# Enable rust backtraces by default for easier debugging
if "RUST_BACKTRACE" not in os.environ:
    os.environ["RUST_BACKTRACE"] = "1"
