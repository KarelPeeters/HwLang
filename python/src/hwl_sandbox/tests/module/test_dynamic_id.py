from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


def test_dynamic_id_pub_wire(tmp_dir: Path):
    src = """
    module top ports(
        clk: in clock,
        rst: in async bool,
        sync(clk, async rst) {
            x: in int(8),
            y: out int(8),
        }
    ) {
        for (i in 0..4) {
            pub wire ident("w{i}");
            comb {
                if (i == 0) { 
                    ident("w{i}") = x;
                } else {
                    ident("w{i}") = ident("w{i-1}");
                }
            }
        }
        
        comb {
            y = w3;
        }
    }
    """
    top: hwl.Module = compile_custom(src).resolve("top.top")
    print(top.as_verilog().source)

    inst = top.as_verilated(tmp_dir).instance()

    inst.ports.x.value = 4
    inst.step(1)
    assert inst.ports.y.value == 4


def test_dynamic_id_ports(tmp_dir: Path):
    src = """
    module top ports(
        async {
            for (i in 0..8) {
                ident("x_{i}"): in int(8), 
                ident("y_{i}"): out int(8), 
            }
        }
    ) {
        comb {
            for (i in 0..8) {
                ident("y_{i}") = ident("x_{i}");
            }
        }
    }
    """
    top: hwl.Module = compile_custom(src).resolve("top.top")
    print(top.as_verilog().source)

    inst = top.as_verilated(tmp_dir).instance()

    for i in range(8):
        inst.ports[f"x_{i}"].value = i
    inst.step(1)
    for i in range(8):
        assert inst.ports[f"y_{i}"].value == i


def test_dynamic_id_interface_view_port_dir():
    src = """
    interface foo(n: uint) {
        for (i in 0..n) {
            ident("x_{i}"): bool,
        }
        interface input {
            for (i in 0..n) {
                ident("x_{i}"): in,
            }
        }
    }
    """
    foo = compile_custom(src).resolve("top.foo")
    foo(n=2)

