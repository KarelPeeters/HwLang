from pathlib import Path

import khdl

from khdl_sandbox.common.util import compile_custom


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
        inst.ports[f'a_{i}'].value = i + 17
    inst.step(1)
    assert [inst.ports[f'b_{i}'].value for i in range(3)] == [19, 18, 17]


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
        inst.ports[f'a_{i}_value'].value = i + 17
    inst.step(1)
    assert [inst.ports[f'b_{i}_value'].value for i in range(3)] == [19, 18, 17]
