from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


def test_dynamic_id(tmp_dir: Path):
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
