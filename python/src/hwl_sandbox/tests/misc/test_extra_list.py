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


def test_extra_list_match():
    src = """
    fn f(c: bool, x: int) -> int {
        match (x) {
            if (c) {
                1 => { return 1; }
            }
            2 => { return 2; }
            _ => { return -x; }  
        }
    }
    """
    f = compile_custom(src).resolve("top.f")

    assert f(True, 1) == 1
    assert f(True, 2) == 2
    assert f(True, 3) == -3

    assert f(False, 1) == -1
    assert f(False, 2) == 2
    assert f(False, 3) == -3


def test_extra_list_module_instance():
    src = """
    module child ports(x: in async bool) {}
    module parent(c: bool) ports() {
        instance child ports(
            x=false,
            if (c) {
                y=true,
            }
        );
    }
    """
    parent = compile_custom(src).resolve("top.parent")

    parent(False)
    with pytest.raises(hwl.DiagnosticException, match="connection does not match any port"):
        parent(True)


def test_extra_list_params():
    src = """
    fn f(c: bool, if (c) { x: int }) -> int {
        if (c) {
            return x;
        } else {
            return 0;
        }
    }
    fn g(c: bool, d: bool, x: int) -> int {
        return f(c, if (d) { x=x });
    }
    """
    g = compile_custom(src).resolve("top.g")

    assert g(False, False, 0) == 0
    assert g(False, False, 1) == 0

    assert g(True, True, 0) == 0
    assert g(True, True, 1) == 1

    with pytest.raises(hwl.DiagnosticException, match="argument did not match"):
        assert g(False, True, 0) == 0
    with pytest.raises(hwl.DiagnosticException, match="missing argument"):
        assert g(True, False, 0) == 0
