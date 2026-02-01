import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_extra_list_interface():
    src = """
    interface foo(X: bool, Y: bool, Z: bool, V: bool) {
        if (X) { x: bool }
        if (Y) { y: bool }

        interface view_0 {
            if (X) { x: in }
            if (Y) { y: out }

            // cause error
            if (Z) { z: out }
        }

        if (V) {
            // cause error
            interface view_1 { d: out }
        }
    }
    """
    foo = compile_custom(src).resolve("top.foo")

    foo(X=False, Y=False, Z=False, V=False)
    foo(X=True, Y=False, Z=False, V=False)
    foo(X=False, Y=True, Z=False, V=False)

    with pytest.raises(hwl.DiagnosticException, match="port not found"):
        foo(X=False, Y=False, Z=True, V=False)
    with pytest.raises(hwl.DiagnosticException, match="port not found"):
        foo(X=False, Y=False, Z=False, V=True)
