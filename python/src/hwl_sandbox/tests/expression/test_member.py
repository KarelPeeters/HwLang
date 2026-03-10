import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_struct_member():
    src = """
    struct Foo {
        x: int,
        
        const C: int = 8;
        fn derp(x: uint) -> uint {
            return x * 2;
        }
    }
    fn f(x: uint) -> int {
       return Foo.C + Foo.derp(x);
    }
    """
    f = compile_custom(src).resolve("top.f")
    assert f(0) == 8
    assert f(4) == 16


def test_struct_member_duplicate():
    src = """
    struct Foo {
        x: int,
        
        const C: int = 8;
        fn derp(x: uint) -> uint { return x * 2; }
        fn derp(x: uint) -> uint { return x * 3; }
    }
    const a = Foo.derp;
    """
    # noinspection PyUnresolvedReferences
    with pytest.raises(
            hwl.DiagnosticException,
            match="identifier `derp` declared multiple times",
            check=lambda e: len(e.messages) == 1
    ):
        a = compile_custom(src).resolve("top.a")
