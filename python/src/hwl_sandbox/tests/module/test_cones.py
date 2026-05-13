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
    with pytest.raises(hwl.DiagnosticException, match="dynamic array assignment to not yet driven signal"):
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
    }
    """
    top = compile_custom(src).resolve("top.top")

    top(c=True)
    with pytest.raises(hwl.DiagnosticException, match="dynamic array assignment to not yet driven signal"):
        top(c=False)


def test_comb_dyn_slice():
    src = """
    module top(c: bool) ports(i: in async int(0..2)) {
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
    with pytest.raises(hwl.DiagnosticException, match="dynamic array assignment to not yet driven signal"):
        top(c=False)
