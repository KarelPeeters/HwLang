import hwl
import pytest

from hwl_sandbox.common.util import compile_custom, diag_error, diag_warning


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
    with diag_error("driver mismatch between conditional branches"):
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
    with diag_error("dynamic array index assignment to not yet driven signal"):
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
    with diag_error("dynamic array index assignment to not yet driven signal"):
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
    with diag_error("dynamic array index assignment to not yet driven signal"):
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
    with diag_error("dynamic array slice assignment to not yet driven signal"):
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
    with diag_warning("wire `w` is not driven"):
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
    with diag_error("combinatorial self-loop"):
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
    with diag_error("wire `w` has multiple overlapping drivers"):
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
    with diag_error("combinatorial self-loop"):
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
    with diag_error("combinatorial self-loop"):
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
    with diag_error("combinatorial self-loop"):
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
    with diag_error("driver mismatch between conditional branches"):
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
    print(e.value)

    # expect two diagnostics, one for each signal
    diags = e.value.diagnostics
    assert len(diags) == 2
    for m in diags:
        assert m.level == "error"
        assert m.title == "driver mismatch between conditional branches"


def test_dyn_array_index_full():
    # test that array cone checking works properly, for every possible combination of indices
    src = """
    module top(N: int, I: [_]bool, J: int) ports() {
        wire w: async [N]bool;
        comb {
            // drive some subset of the wire
            for (i in 0..N) {
                if (I[i]) {
                    w[i] = false;
                }
            }
            // read the entire wire
            val v = w;
            // drive some extra index, this should only be accepted if we already drove it earlier
            w[J] = true;
        }
        comb {
            // drive the rest of the bits to supress the warning
            for (i in 0..N) {
                if (!I[i]) {
                    w[i] = false;
                }
            }
        }
    }
    """
    n = 4

    any_valid = False
    any_invalid = False
    for i_mask in range(2 ** n):
        i_array = [i_mask & (1 << i) != 0 for i in range(n)]
        for j in range(n):
            valid = i_array[j]

            top = compile_custom(src).resolve("top.top")
            if valid:
                _ = top(N=n, I=i_array, J=j)
                any_valid = True
            else:
                with diag_error("combinatorial self-loop"):
                    _ = top(N=n, I=i_array, J=j)
                any_invalid = True

    assert any_valid and any_invalid


def test_dyn_array_index_slice_full():
    # test that array cone checking works properly, for every possible combination of indices
    src = """
    module top(N: int, M: int, I: [_]bool, J: int) ports() {
        wire w: async [N]bool;
        comb {
            // drive some subset of the wire
            for (i in 0..N) {
                if (I[i]) {
                    w[i] = false;
                }
            }
            // read the entire wire
            val v = w;
            // drive some extra slice, this should only be accepted if we already drove all involved wires earlier
            w[J+..M] = [true] * M;
        }
        comb {
            // drive the rest of the bits to supress the warning
            for (i in 0..N) {
                if (!I[i]) {
                    w[i] = false;
                }
            }
        }
    }
    """
    n = 4
    m = 2

    any_valid = False
    any_invalid = False
    for i_mask in range(2 ** n):
        i_array = [i_mask & (1 << i) != 0 for i in range(n)]
        for j in range(n - m):
            valid = all(i_array[j:j + m])

            top = compile_custom(src).resolve("top.top")
            if valid:
                _ = top(N=n, M=m, I=i_array, J=j)
                any_valid = True
            else:
                with diag_error("combinatorial self-loop"):
                    _ = top(N=n, M=m, I=i_array, J=j)
                any_invalid = True

    assert any_valid and any_invalid


def test_dyn_array_huge():
    # test that huge arrays can be tracked efficiently (thanks to the sparse array mask data structure)
    src = """
    const N = 2**128;
    module top(c: bool) ports(x: in async [N/2]bool, y: out async [N]bool) {
        comb {
            y[..N/2] = x;
            if (c) {
                y[N/2..] = y[..N/2];
            } 
        }
    }
    """
    top = compile_custom(src).resolve("top.top")

    _ = top(c=True)
    with diag_warning("port `y` is not fully driven"):
        _ = top(c=False)


def test_cones_multiple_drivers_paths():
    src = """
    struct Foo {
        array: [4][4]bool,
        tuple: Tuple(bool, uint(8)),
        tuple_nested: Tuple(Tuple(bool, bool)),
        compound_overdriven: Tuple(bool, bool),
    }
    pub module top ports() {
        wire w: Foo;
        comb {
            w.array = [[false] * 4] * 4;
            w.tuple = (false, 0);
            w.tuple_nested = ((false, false),);
            w.compound_overdriven = (false, false);
        }
        comb {
            w.array[0][..2] = [true, true];
            w.array[1][2..] = [true, true];
            w.tuple.1 = 4;
            w.tuple_nested.0.0 = true;
            w.compound_overdriven = (false, false);
        }
    }
    """
    expected_paths = [
        "w.array[0][0..2]",
        "w.array[1][2..4]",
        "w.tuple.1",
        "w.tuple_nested.0.0",
        "w.compound_overdriven",
    ]

    # with diag_error("wire `w` has multiple overlapping drivers") as e:
    with pytest.raises(hwl.DiagnosticException) as e:
        compile_custom(src).resolve_module("top.top")

    diags = e.value.diagnostics
    assert len(diags) == 1
    diag = diags[0]

    assert diag.level == "error"
    for path in expected_paths:
        assert path in diag.full_string


def test_cones_mix_driven_undriven_overdriven():
    src = """
    const zero = 0;
    const one = 1;
    module top ports() {
        wire w: [3]bool;
        comb {
            w[zero] = true;
            w[one] = true;
        }
        comb {
            w[zero] = true;
        }
    }
    """
    with pytest.raises(hwl.DiagnosticException) as e:
        compile_custom(src).resolve_module("top.top")

    print(e.value)

    diags = e.value.diagnostics
    assert len(diags) == 2

    assert diags[0].level == "error"
    assert diags[0].title == "wire `w` has multiple overlapping drivers"
    assert "w[0]" in diags[0].full_string

    assert diags[1].level == "warning"
    assert diags[1].title == "wire `w` is not fully driven"
    assert "w[2]" in diags[1].full_string

    for diag in diags:
        assert "w[1]" not in diag.full_string


def test_cones_clocked_dyn_index():
    src = """
    module top ports(clk: in clock, sync(clk) { x: in [4]bool, i: in int(0..4) }) {
        clocked(clk) {
            reg r: [4]bool = undef;
            r = x;
            val y: bool = r[i];
        }
    }
    """
    _ = compile_custom(src).resolve_module("top.top")
