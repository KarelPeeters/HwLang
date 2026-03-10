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
