import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


# TODO extensive tests for mixes of full/partial/dynamic writes

def test_comb_assign_conditional():
    src = """
    module top(c_root: bool, c_else: bool) ports(b: in async bool) {
        wire w: async bool;
        comb {
            if (c_root) {
                w = true;
            }
        
            if (b) {
                w = false;
            } else {
                if (c_else) {
                    w = true;
                }            
            }
            
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    top(c_root=True, c_else=False)
    top(c_root=False, c_else=True)
    top(c_root=True, c_else=True)
    with pytest.raises(hwl.DiagnosticException, match="driver mismatch between conditional branches"):
        top(c_root=False, c_else=False)


def test_comb_dyn_index():
    src = """
    module top(c: bool) ports(i: in async int(0..2)) {
        wire w: async [4]bool;
        comb {
            if (c) {
                w = [false] * 4;
            }
            w[i] = false;
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    top(c=True)
    with pytest.raises(hwl.DiagnosticException, match="dynamic array index assignment to not yet driven signal"):
        top(c=False)


def test_comb_dyn_index_partial():
    src = """
    module top(c: bool) ports(i: in async int(0..2)) {
        wire w: async [4]bool;
        comb {
            if (c) {
                w[0..2] = [false] * 2;
            }
            w[i] = false;
        }
        comb {
            w[2..] = [false] * 2;
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    top(c=True)
    with pytest.raises(hwl.DiagnosticException, match="dynamic array index assignment to not yet driven signal"):
        top(c=False)


def test_comb_dyn_index_tuple_field_after_field_default():
    src = """
    module top ports(i: in async int(0..4)) {
        wire w: async [4]Tuple(bool, bool);
        comb {
            for (j in 0..4) {
                w[j].0 = false;
            }
            w[i].0 = true;
            for (j in 0..4) {
                w[j].1 = false;
            }
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_dyn_index_tuple_field_before_field_default():
    src = """
    module top ports(i: in async int(0..4)) {
        wire w: async [4]Tuple(bool, bool);
        comb {
            for (j in 0..4) {
                w[j].1 = false;
            }
            w[i].0 = true;
            for (j in 0..4) {
                w[j].0 = false;
            }
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="dynamic array index assignment to not yet driven signal"):
        compile_custom(src).resolve_module("top.top")


def test_comb_dyn_slice():
    src = """
    module top(c: bool) ports(i: in async int(0..3)) {
        wire w: async [4]bool;
        comb {
            if (c) {
                w = [false] * 4;
            }
            w[i+..2] = [false] * 2;
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    top(c=True)
    with pytest.raises(hwl.DiagnosticException, match="dynamic array slice assignment to not yet driven signal"):
        top(c=False)


def test_comb_dyn_empty_slice_before_default():
    src = """
    module top ports(i: in async int(0..5)) {
        wire w: async [4]bool;
        comb {
            w[i+..0] = [];
            w = [false] * 4;
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_dyn_empty_slice_does_not_drive():
    src = """
    module top ports(i: in async int(0..5)) {
        wire w: async [4]bool;
        comb {
            w[i+..0] = [];
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has no driver"):
        compile_custom(src).resolve_module("top.top")


def test_comb_write_after_read():
    src = """
    module top ports(x: in async bool, y: out async bool) {
        wire w: async bool;
        comb {
            y = w;
            w = x; 
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="combinatorial self-loop"):
        compile_custom(src).resolve_module("top.top")


def test_comb_write_after_read_non_overlapping():
    src = """
    module top ports(x: in async bool, y: out async bool) {
        wire w: async [2]bool;
        comb {
            y = w[0];
            w[1] = x; 
        }
        comb {
            w[0] = x;
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_multi_process_partial_drivers_non_overlapping():
    src = """
    module top ports(x: in async bool) {
        wire w: async [2]bool;
        comb {
            w[0] = x;
        }
        comb {
            w[1] = x;
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_multi_process_partial_drivers_overlapping():
    src = """
    module top ports(x: in async bool) {
        wire w: async [2]bool;
        comb {
            w[0] = x;
        }
        comb {
            w[0] = x;
            w[1] = x;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="wire `w` has multiple overlapping drivers"):
        compile_custom(src).resolve_module("top.top")


def test_comb_read_after_write():
    src = """
    module top ports(x: in async bool, y: out async bool) {
        wire w: async bool;
        comb {
            w = x;
            y = w; 
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_dynamic_read_after_all_possible_writes():
    src = """
    module top ports(x: in async bool, y: out async bool, i: in async int(0..2)) {
        wire w: async [2]bool;
        comb {
            w[0] = x;
            w[1] = x;
            y = w[i];
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_dynamic_read_before_some_possible_write():
    src = """
    module top ports(x: in async bool, y: out async bool, i: in async int(0..2)) {
        wire w: async [2]bool;
        comb {
            w[0] = x;
            y = w[i];
            w[1] = x;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="combinatorial self-loop"):
        compile_custom(src).resolve_module("top.top")


def test_comb_slice_then_index_read_after_write():
    src = """
    module top ports(i: in async int(0..3), y: out async bool) {
        wire w: async [4]bool;
        comb {
            w = [false] * 4;
            y = (w[i+..2])[0];
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_slice_then_index_read_before_write():
    src = """
    module top ports(i: in async int(0..3), y: out async bool) {
        wire w: async [4]bool;
        comb {
            y = (w[i+..2])[0];
            w = [false] * 4;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="combinatorial self-loop"):
        compile_custom(src).resolve_module("top.top")


def test_comb_read_slice_then_index():
    src = """
    module top ports(i: in async int(0..3), y: out async bool) {
        wire w: async [4]bool;
        comb {
            y = w[i+..2][0];
            w = [false] * 4;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="combinatorial self-loop"):
        compile_custom(src).resolve_module("top.top")


def test_comb_multi_step_cone():
    src = """
    module top(c: bool) ports(x: in async bool) {
        wire w: async [4]Tuple(bool, uint(8));
        comb {
            for (i in 0..4) { w[i].0 = false; }
            
            if (x) {
                if (c) {
                    w[0].0 = true;
                } else {
                    w[0].1 = 1;
                }
            } 
        }
        // drive the rest of the wire
        comb {
            for (i in 0..4) {
                if (c) {
                    w[i].1 = 0;
                } else {
                    w[i].0 = false;
                }
            }
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    top(c=True)
    with pytest.raises(hwl.DiagnosticException, match="driver mismatch between conditional branches"):
        top(c=False)


def test_clocked_assign_conditional():
    src = """
    module top ports(clk: in clock, sync(clk) { c: in bool, x: in bool }) {
        clocked(clk) {
            reg r: bool = undef;
            if (c) {
                r = x;
            } 
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_clocked_assign_dyn_index():
    src = """
    module top ports(clk: in clock, sync(clk) { i: in int(0..4) }) {
        clocked(clk) {
            reg r: [4]bool = undef;
            r[i] = false; 
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_clocked_assign_dyn_slice():
    src = """
    module top ports(clk: in clock, sync(clk) { i: in int(0..3) }) {
        clocked(clk) {
            reg r: [4]bool = undef;
            r[i+..2] = [false, false]; 
        }
    }
    """
    compile_custom(src).resolve_module("top.top")


def test_comb_assign_conditional_multiple_signals_no_duplicates():
    # check that we only get a single error per signal, but do get errors for all signals
    src = """
    module top ports(b: in async bool, c: in async bool) {
        wire v: async bool;
        wire w: async bool;
        comb {
            if (b) {
                if (c) {
                    w = true;
                }
                v = true;
            }
        }
    }
    """

    with pytest.raises(hwl.DiagnosticException) as e:
        _ = compile_custom(src).resolve("top.top")

    messages = e.value.messages
    assert len(messages) == 2
    for m in messages:
        assert "driver mismatch between conditional branches" in m
