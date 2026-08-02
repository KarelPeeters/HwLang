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


# TODO fix this deadlock by moving all elaboration into a single loop-detecting data structure
@pytest.mark.skip
def test_cycle_struct_recurse_generic():
    with diag_error("cyclic dependency"):
        c = compile_custom("struct S(T: type) { a: int, b: S(T) }")
        s = c.resolve("top.S")
        _ = s(int)


# TODO fix this deadlock by moving all elaboration into a single loop-detecting data structure
def test_cycle_module_header():
    src = """
    module top ports(
        const a = top;
    ) {}
    """
    c = compile_custom(src)
    _ = c.resolve(f"top.top")


@pytest.mark.skip
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

# TODO add test for cycle that mixes things:
#    item that depends on (generic) struct that depends on item again
