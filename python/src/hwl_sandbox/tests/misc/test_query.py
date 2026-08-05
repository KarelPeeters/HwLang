import pytest

from hwl_sandbox.common.util import compile_custom, diag_error


def test_cycle_constants():
    src = """
    const a = b;
    const b = a;
    """

    c = compile_custom(src)
    with diag_error("encountered cyclic dependency"):
        c.resolve("top.a")


def test_cycle_struct_recurse_simple():
    c = compile_custom("struct S { a: int, b: S }")
    with diag_error("encountered cyclic dependency"):
        _ = c.resolve("top.S")


@pytest.mark.xfail(run=False)
def test_cycle_struct_recurse_generic():
    with diag_error("encountered cyclic dependency"):
        c = compile_custom("struct S(T: type) { a: int, b: S(T) }")
        s = c.resolve("top.S")
        _ = s(int)


@pytest.mark.xfail(run=False)
def test_cycle_module_header():
    src = """
    module top ports(
        const a = top;
    ) {}
    """
    c = compile_custom(src)
    with diag_error("encountered cyclic dependency"):
        _ = c.resolve("top.top")


@pytest.mark.xfail(run=False)
def test_cycle_mixed():
    src = """
    const const_a = type_b;
    
    type type_b = struct_c(false);
    
    struct struct_c(b: bool) {
        const _ = module_d;
    }
    
    module module_d ports(
        const _ = const_a;
    ) {}
    """
    c = compile_custom(src)
    with diag_error("encountered cyclic dependency") as e:
        _ = c.resolve("top.const_a")

    # TODO: get indices to pairwise match
    # TODO: replace "function declared here" with "generic item declared here" if applicable
    expected_messages = [
        "[0] item declared here",
        "[1] item used here",
        "[2] item declared here",
        "[3] function call here",
        "[4] function declared here",
        "[5] item used here",
        "[6] item declared here",
        "[7] item used here",
    ]
    assert e.diag.messages == expected_messages


@pytest.mark.xfail(run=False)
def test_chain_struct():
    src = """
    struct S0 {}
    """
    n = 128
    for i in range(n):
        src += f"struct S{i + 1} {{ f: S{i} }}"

    c = compile_custom(src)
    s = c.resolve(f"top.S{n}")
    print(s)
