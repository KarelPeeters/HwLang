import os


# In a separate file that does not import hwl, to avoid importing before re-installing the module.
def enable_rust_backtraces():
    var = "RUST_BACKTRACE"
    if var not in os.environ:
        os.environ[var] = "1"
