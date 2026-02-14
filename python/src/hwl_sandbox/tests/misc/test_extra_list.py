import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_extra_list_in_interface():
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


def test_extra_list_in_match():
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


def test_extra_list_in_module_ports():
    src = """
    module top(c: bool) ports(
        x0: in async bool,
        if (c) {
            y0: in async bool,
        }
        
        async {
            x1: in bool,
            if (c) {
                y1: in bool,
            }            
        }
        
        if (c) {
            async {
                y2: in bool,
            }
        }
    ) {
        comb {
            val _ = x0;
            val _ = x1;
            if (c) {
                val _ = y0;
                val _ = y1;
                val _ = y2;
            }
        }
    }
    """

    top = compile_custom(src).resolve("top.top")
    top(c=False)
    top(c=True)


def test_extra_list_in_module_instance():
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


def test_extra_list_in_params():
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


def test_extra_list_params_conflict():
    src = "fn f(const c = false; c: bool) {}"
    f = compile_custom(src).resolve("top.f")
    with pytest.raises(hwl.DiagnosticException, match="declared multiple times"):
        f(c=False)


def test_extra_list_scopes():
    # TODO also test this for modules, where scopes are restored later
    # TODO test for loops
    src = """
    fn f(
        c: bool,
        d: bool,
        e: bool,
        
        // top-level scope is visible everywhere, including the function body
        const A = 4;
        
        if (c) {
            // branches have local, non-conflicting scopes
            const B = 4;
            b: uint(B),
            
            const C = 8;
        } else {
            const B = 2;
            b: uint(B),
        }
        
        if (e) {
            // we can reuse names in different branches
            const B = 4;
        }
        if (d) {
            // branches scopes don't leak outside
            const { print(C); } 
        }
    ) -> int {
        return A;
    }
    """
    f = compile_custom(src).resolve("top.f")

    # top-level visible
    assert f(c=True, d=False, e=False, b=0) == 4

    # branches local
    f(c=True, d=False, e=False, b=2 ** 4 - 1)
    f(c=True, d=False, e=False, b=2 ** 2 - 1)
    with pytest.raises(hwl.DiagnosticException, match="type mismatch"):
        f(c=False, d=False, e=False, b=2 ** 4 - 1)
    f(c=False, d=False, e=False, b=2 ** 2 - 1)

    # reuse
    f(c=True, d=False, e=True, b=0)

    # no leaking
    with pytest.raises(hwl.DiagnosticException, match="undeclared identifier"):
        f(c=True, d=True, e=False, b=0)
    with pytest.raises(hwl.DiagnosticException, match="undeclared identifier"):
        f(c=False, d=True, e=False, b=0)


def test_extra_item_kind_decl():
    src = """
    fn f(const x: int = 4;) -> int { return x; }
    """
    f = compile_custom(src).resolve("top.f")
    assert f() == 4


def test_extra_item_kind_if():
    src = """
    fn f(if (true) { x: int } else { y: int }) -> int { return x; }
    """
    f = compile_custom(src).resolve("top.f")
    assert f(x=4) == 4


def test_extra_item_kind_for():
    src = """
    fn f(for (i in 0..=1) { x: int }) -> int { return x; }
    """
    f = compile_custom(src).resolve("top.f")
    assert f(x=4) == 0


def test_extra_item_kind_match():
    src = """
    fn f(match (0) { 0 => { x: int }) -> int { return x; }
    """
    f = compile_custom(src).resolve("top.f")
    assert f(x=4) == 0
