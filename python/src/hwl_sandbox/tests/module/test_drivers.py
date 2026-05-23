import pytest

import hwl
from hwl_sandbox.common.util import compile_custom, diag_error, diag_warning


def test_port_driver_none():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
    }
    """
    with diag_warning("port `y` is not driven"):
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
    with diag_error("port `y` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_port_driver_multi_comb_clocked():
    src = """
    module top ports(clk: in clock, rst: in async bool, y: out sync(clk, async rst) bool) {
        comb {
            y = false;
        }
        clocked(clk, async rst) {
            reg wire y = false;
        }
    }
    """
    with diag_error("port `y` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_none_with_type():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: bool;
    }
    """
    with diag_warning("wire `w` is not driven"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_none_with_unit_type():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: Tuple();
    }
    """
    with diag_warning("wire `w` is not driven"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_none_without_type():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w;
    }
    """
    with diag_warning("wire `w` is not driven"):
        _ = compile_custom(src).resolve("top.top")


def test_wire_driver_single_expr_unit_type():
    src = """
    module top ports(clk: in clock, rst: in async bool) {
        wire w: Tuple() = ();
    }
    """
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
    module top ports(clk: in clock, rst: in async bool) {
        wire w: bool;
        comb {
            w = false;
        }
        comb {
            w = true;
        }
    }
    """
    with diag_error("wire `w` has multiple drivers"):
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
    with diag_error("wire `w` has multiple drivers"):
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
    with diag_error("wire `w` has multiple drivers"):
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
    with diag_error("wire `w` has multiple drivers"):
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
    with diag_error("wire `w` has multiple drivers"):
        _ = compile_custom(src).resolve("top.top")


def test_write_to_signal_in_wire_expr():
    src = """
    module top ports(x: in async bool, y: out async bool) {
        wire w = {
            y = false;
            x
        };
    }
    """
    with diag_error("assigning to signals is only allowed in processes"):
        _ = compile_custom(src).resolve("top.top")


def test_write_to_signal_in_port_expr():
    src = """
    module top ports(x: in async bool, y: out async bool) {
        instance child ports(x={
            y = false;
            x
        });
    }
    module child ports(x: in async bool) {}
    """
    with diag_error("assigning to signals is only allowed in processes"):
        _ = compile_custom(src).resolve("top.top")
