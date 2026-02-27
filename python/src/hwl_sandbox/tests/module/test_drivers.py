import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_port_driver_none():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="port `y` has no driver"):
        _ = compile_custom(src).resolve("top.top")


def test_port_driver_single_comb():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        comb {
            y = false;
        }
    }
    """
    _ = compile_custom(src).resolve("top.top")


def test_port_driver_multi_comb():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        comb {
            y = false;
        }
        comb {
            y = true;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="port `y` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_port_driver_multi_comb_clocked():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        comb {
            y = false;
        }
        clocked(clk, async rst) {
            reg port y = false;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="port `y` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_none_with_type():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: bool;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has no driver"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_none_without():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w;
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has no driver"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_single_expr():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: bool = false;
    }
    """
    _ = compile_custom(src).resolve("top.top")


def test_wire_driver_single_comb():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: bool;
        comb {
            w = false;
        }
    }
    """
    _ = compile_custom(src).resolve("top.top")


def test_wire_driver_multi_comb():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        wire w: bool;
        comb {
            w = false;
        }
        comb {
            w = true;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_multi_expr_comb():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: bool = false;
        comb {
            w = false;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_multi_expr_clocked_with_assign():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: sync(clk, async rst) bool = false;
        clocked(clk, async rst) {
            reg wire w = false;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_multi_expr_clocked_without_assign():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: sync(clk, async rst) bool = false;
        clocked(clk, async rst) {
            reg wire w = false;
            w = !w;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_multi_expr_child():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: async bool = false;
        instance child ports(y=w);
    }
    module child ports(y: out async bool) {
        comb {
            y = false;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")
