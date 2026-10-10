import pytest

from hwl_sandbox.common.util import compile_custom, diag_error


def test_nested_struct_fresh_contexts():
    # Resolving Outer must elaborate Inner recursively. Random cache hashing can
    # place both structs in the same shard; that must not deadlock elaboration.
    for _ in range(512):
        c = compile_custom("struct Inner { value: uint } struct Outer { inner: Inner }")
        _ = c.resolve("top.Outer")


# TODO fix this deadlock by moving all elaboration into a single loop-detecting data structure
@pytest.mark.skip
def test_type_recursive_struct_generic():
    with diag_error("cyclic dependency"):
        c = compile_custom("struct S(T: type) { a: int, b: S(T) }")
        s = c.resolve("top.S")
        _ = s(int)
