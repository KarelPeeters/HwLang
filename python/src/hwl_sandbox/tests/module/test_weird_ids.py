from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


def test_module_duplicate_child_id(tmp_dir: Path):
    src = """
    module top ports() {
        instance my_name = child ports();
        instance my_name = child ports();
    }
    module child ports() {}
    
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")

    print(top.as_verilog().source)
    _ = top.as_verilated(tmp_dir)


def test_module_collide_port_child(tmp_dir: Path):
    src = """
    module top ports(my_name: in async bool) {
        instance my_name = child ports();
    }
    module child ports() {}
    
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")

    print(top.as_verilog().source)
    _ = top.as_verilated(tmp_dir)


def test_module_collide_port_wire(tmp_dir: Path):
    src = """
    module top ports(
        my_name: in async bool,
        y: out async bool,
    ) {
        wire my_name = my_name;
        comb { y = my_name; }
    }
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")

    print(top.as_verilog().source)
    inst = top.as_verilated(tmp_dir).instance()

    for v in [False, True]:
        inst.ports.my_name.value = v
        inst.step(1)
        assert inst.ports.y.value == v


def test_module_collide_port_process(tmp_dir: Path):
    src = """
    module top ports(
        clk: in clock,
        
        comb_0: in async bool,
        comb_1: in async bool,
        clocked_0: in async bool,
        clocked_1: in async bool,
    ) {
        comb {
            print("test comb");
        }
        clocked(clk) {
            print("test clocked");
        }
    }
    """
    top = compile_custom(src).resolve("top.top")
    verilog = top.as_verilog().source
    print(verilog)
    _ = top.as_verilated(tmp_dir)

    # check that we did indeed push the processes to different names
    assert "always @(*) begin: comb_2" in verilog
    assert "always @(posedge clk) begin: clocked_2" in verilog


def test_module_internal_name_keyword(tmp_dir: Path):
    src = """
    module top ports() {
        instance input ports();
    }
    module input ports() {}
    """

    top = compile_custom(src).resolve("top.top")
    _ = top.as_verilated(tmp_dir)


def test_module_external_name_keyword(tmp_dir: Path):
    src = """
    external module input ports()
    module top ports() {
        instance input ports();
    }
    
    """
    top = compile_custom(src).resolve("top.top")
    extra_verilog_files = [Path(__file__).parent / "external.v"]
    _ = top.as_verilated(tmp_dir, extra_verilog_files=extra_verilog_files)
