from pathlib import Path

import pytest

import khdl

from khdl_sandbox.common.util import compile_custom, diag_error


def test_instance_port_loop_scope(tmp_dir: Path):
    src = '''
    module Copy ports(async {
        for (i in 0..3) {
            ident("x_{i}"): in uint(8),
            ident("y_{i}"): out uint(8),
        }
    }) {
        comb {
            for (i in 0..3) { ident("y_{i}") = ident("x_{i}"); }
        }
    }
    module Top ports(async {
        for (i in 0..3) {
            ident("a_{i}"): in uint(8),
            ident("b_{i}"): out uint(8),
        }
    }) {
        instance Copy ports(
            for (i in 0..3) {
                ident("x_{i}")=ident("a_{2-i}"),
                ident("y_{i}")=ident("b_{i}"),
            }
        );
    }
    '''
    module: khdl.Module = compile_custom(src).resolve('top.Top')
    inst = module.as_verilated(tmp_dir).instance()
    for i in range(3):
        inst.ports[f'a_{i}'] = i + 17
    inst.step(1)
    assert [inst.ports[f'b_{i}'] for i in range(3)] == [19, 18, 17]


def test_instance_interface_connection_loop_scope(tmp_dir: Path):
    src = '''
    interface Data {
        value: uint(8),
        view Source { value: out }
        view Sink { value: in }
    }
    module Copy ports(async {
        for (i in 0..3) {
            ident("x_{i}"): interface Data.Sink,
            ident("y_{i}"): interface Data.Source,
        }
    }) {
        comb {
            for (i in 0..3) { ident("y_{i}").value = ident("x_{i}").value; }
        }
    }
    module Top ports(async {
        for (i in 0..3) {
            ident("a_{i}"): interface Data.Sink,
            ident("b_{i}"): interface Data.Source,
        }
    }) {
        instance Copy ports(
            for (i in 0..3) {
                ident("x_{i}")=ident("a_{2-i}"),
                ident("y_{i}")=ident("b_{i}"),
            }
        );
    }
    '''
    module: khdl.Module = compile_custom(src).resolve('top.Top')
    inst = module.as_verilated(tmp_dir).instance()
    for i in range(3):
        inst.ports[f'a_{i}'].value = i + 17
    inst.step(1)
    assert [inst.ports[f'b_{i}'].value for i in range(3)] == [19, 18, 17]


def test_instance_port_loop_domain_reversed_order():
    src = """
    module Reg ports(
        for (i in 0..2) {
            ident("clk_{i}"): in clock,
            ident("x_{i}"): in sync(ident("clk_{i}")) bool,
            ident("y_{i}"): out sync(ident("clk_{i}")) bool,
        }
    ) {
        comb {
            for (i in 0..2) {
                ident("y_{i}") = ident("x_{i}");
            }
        }
    }
    module Top ports(
        for (i in 0..2) {
            ident("c_{i}"): in clock,
            ident("a_{i}"): in sync(ident("c_{i}")) bool,
            ident("b_{i}"): out sync(ident("c_{i}")) bool,
        }
    ) {
        wire w_0: bool;
        wire w_1: bool;
        instance Reg ports(
            for (j in 0..2) {
                ident("y_{1-j}")=ident("w_{1-j}"),
                ident("x_{1-j}")=ident("a_{1-j}"),
                ident("clk_{1-j}")=ident("c_{1-j}"),
            }
        );
        comb {
            b_0 = w_0;
            b_1 = w_1;
        }
    }
    """
    _ = compile_custom(src).resolve("top.Top")


def test_instance_port_loop_domain_mismatch():
    src = """
    module Reg ports(
        for (i in 0..2) {
            ident("clk_{i}"): in clock,
            ident("x_{i}"): in sync(ident("clk_{i}")) bool,
        }
    ) {}
    module Top ports(
        for (i in 0..2) {
            ident("c_{i}"): in clock,
            ident("a_{i}"): in sync(ident("c_{i}")) bool,
        }
    ) {
        instance Reg ports(
            for (i in 0..2) {
                ident("x_{i}")=ident("a_{i}"),
                ident("clk_{i}")=ident("c_{1-i}"),
            }
        );
    }
    """
    with pytest.raises(khdl.DiagnosticException) as e:
        _ = compile_custom(src).resolve("top.Top")
    titles = [d.title for d in e.value.diagnostics]
    assert titles == ["invalid domain crossing: different clock"] * 2


def test_instance_duplicate_connections_single():
    src = """
    module Child ports(x: in async bool, y: in async bool) {}
    module Top ports(a: in async bool) {
        instance Child ports(x=a, y=a, x=a, for (i in 0..2) { x=a });
    }
    """
    with diag_error("multiple connections for port `x`"):
        _ = compile_custom(src).resolve("top.Top")


def test_instance_duplicate_connections_multiple():
    src = """
    module Child ports(x: in async bool, y: in async bool) {}
    module Top ports(a: in async bool) {
        instance Child ports(x=a, y=a, x=a, y=a, z=a, z=a);
    }
    """
    with pytest.raises(khdl.DiagnosticException) as e:
        _ = compile_custom(src).resolve("top.Top")
    titles = sorted(d.title for d in e.value.diagnostics)
    assert titles == [
        "connection does not match any port",
        "connection does not match any port",
        "multiple connections for port `x`",
        "multiple connections for port `y`",
    ]


def test_instance_connection_errors_all_reported():
    src = """
    module Child ports(x: in async bool, y: in async bool, z: in async bool, w: in async bool) {}
    module Top ports(a: in async bool) {
        instance Child ports(x=undefined_0, y=a, z=undefined_1, v=a);
    }
    """
    with pytest.raises(khdl.DiagnosticException) as e:
        _ = compile_custom(src).resolve("top.Top")
    titles = sorted(d.title for d in e.value.diagnostics)
    assert titles == [
        "connection does not match any port",
        "missing connection for port `w`",
        "undeclared identifier `undefined_0`",
        "undeclared identifier `undefined_1`",
    ]


def test_instance_connection_error_in_domain_port():
    src = """
    module Child ports(clk: in clock, x: in sync(clk) bool, y: in async bool) {}
    module Top ports(c: in clock, a: in sync(c) bool, b: in async bool) {
        instance Child ports(x=a, clk=undefined, y=b);
    }
    """
    with diag_error("undeclared identifier `undefined`"):
        _ = compile_custom(src).resolve("top.Top")


def test_instance_duplicate_connection_evaluated():
    src = """
    module Child ports(x: in async bool) {}
    module Top ports(a: in async bool) {
        instance Child ports(x=a, x=undefined);
    }
    """
    with pytest.raises(khdl.DiagnosticException) as e:
        _ = compile_custom(src).resolve("top.Top")
    titles = sorted(d.title for d in e.value.diagnostics)
    assert titles == [
        "multiple connections for port `x`",
        "undeclared identifier `undefined`",
    ]
