from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.util import compile_custom, diag_error


def test_interact_add():
    c = compile_custom("fn f(a: int, b: int) -> int { return a + b; }")
    f = c.resolve("top.f")
    assert f(3, 4) == 7


def test_interact_string_array():
    c = compile_custom("fn f(a: [3]str, b: int) -> str { return a[b]; }")
    f = c.resolve("top.f")
    assert f(["a", "b", "c"], 1) == "b"


def test_interact_types():
    c = compile_custom("fn f(T: type, x: T) -> T { return x; }")
    f = c.resolve("top.f")

    assert f(bool, False) is False
    assert f(bool, True) is True
    with diag_error("type mismatch"):
        f(bool, 0)

    assert f(int, 0) == 0
    with diag_error("type mismatch"):
        f(int, False)


def test_reject_cross_compile_values():
    c0 = compile_custom("struct Foo {}")
    t = c0.resolve("top.Foo")
    c1 = compile_custom("fn f(T: type) -> uint { return T.size_bits; }")
    f = c1.resolve("top.f")

    with pytest.raises(ValueError, match="cannot mix values from different Compile instances"):
        f(t)


def test_compile_manifest():
    manifest_path = Path(__file__).parent / "project/hwl.toml"
    s = hwl.Source.new_from_manifest_path(str(manifest_path))
    c = s.compile()
    assert isinstance(c.resolve("top.top"), hwl.Module)


def test_format():
    src = "const c = a+b;"
    expected = "const c = a + b;\n"
    assert hwl.format_file(src) == expected


def test_capture_prints():
    src = """
    fn f(a: int) -> int {
        print("hello");
        print("world");
        return a + 1;
    }
    """
    c = compile_custom(src)
    f = c.resolve("top.f")

    with c.capture_prints() as capture:
        result = f(5)

    assert result == 6
    assert capture.prints == ["hello\n", "world\n"]


def test_call_type():
    c = compile_custom("")
    uint = c.resolve("std.types.uint")
    assert str(uint) == "uint"
    assert str(uint(8)) == "int(0..256)"


def test_interact_struct(tmp_dir: Path):
    src = """
    struct Pair { x: uint(8), y: bool }
    fn f(x: uint(8), y: bool) -> Pair {
        return Pair.new(x=x, y=y);
    }
    """

    c = compile_custom(src)
    pair = c.resolve("top.Pair")
    f = c.resolve("top.f")

    # check struct construction, indirectly and directly
    assert str(f(4, False)) == "Pair.new(x=4, y=false)"
    assert str(pair.new(x=4, y=False)) == "Pair.new(x=4, y=false)"

    # check struct equality
    a0 = f(4, False)
    a1 = f(4, False)
    b = f(5, False)
    assert a0 == a0
    assert a0 == a1
    assert not (a0 == b)
    assert not (a0 != a1)
    assert a0 != b

    # check comparing values python values works as expected
    assert not (a0 == "test")
    assert a0 != "test"

    # check that we get normal python behavior for non-existing attributes
    with pytest.raises(AttributeError):
        _ = f(4, False).non_existing


def test_interact_enum():
    src = """
    enum Foo { Empty, Data(uint(8)) }
    fn f(x: bool, y: uint(8)) -> Foo {
        if (x) {
            return Foo.Data(y);
        } else {
            return Foo.Empty;
        }
    }
    """

    c = compile_custom(src)
    foo = c.resolve("top.Foo")
    f = c.resolve("top.f")

    # check enum construction, indirectly and directly
    assert str(f(False, 0)) == "Foo.Empty"
    assert str(f(True, 0)) == "Foo.Data(0)"
    assert str(f(True, 1)) == "Foo.Data(1)"

    assert str(foo.Empty) == "Foo.Empty"
    assert str(foo.Data(0)) == "Foo.Data(0)"
    assert str(foo.Data(1)) == "Foo.Data(1)"


def _verilated_port_test_module(tmp_dir: Path) -> hwl.VerilatedInstance:
    src = """
    module top ports(
        x: in async bool,
        y: out async bool,
        n: in async uint(4),
    ) {
        comb { y = x; }
    }
    """
    top: hwl.Module = compile_custom(src).resolve("top.top")
    return top.as_verilated(tmp_dir).instance()


def test_port_interaction_errors(tmp_dir: Path):
    inst = _verilated_port_test_module(tmp_dir)

    # assigning a port directly should point at the correct `.value` syntax
    with pytest.raises(ValueError, match=r"cannot set port value directly, use `ports\.x\.value = value`"):
        inst.ports.x = True
    with pytest.raises(ValueError, match=r'cannot set port value directly, use `ports\["x"\]\.value = value`'):
        inst.ports["x"] = True
    with pytest.raises(ValueError, match=r"cannot set port value directly, use `ports\.y\.value = value`"):
        inst.ports.y = True
    with pytest.raises(ValueError, match=r'cannot set port value directly, use `ports\["y"\]\.value = value`'):
        inst.ports["y"] = True

    # accessing a port that does not exist is a plain attribute error
    with pytest.raises(AttributeError, match="port `missing` not found"):
        _ = inst.ports.missing
    with pytest.raises(AttributeError, match="port `missing` not found"):
        _ = inst.ports["missing"]

    # a port object is not a bool, reading a boolean port requires `.value`
    with pytest.raises(ValueError, match="cannot be used as a boolean"):
        bool(inst.ports.x)
    with pytest.raises(ValueError, match="cannot be used as a boolean"):
        if inst.ports.x:
            pass

    # setting a value of the wrong type is reported as a normal compiler diagnostic
    with diag_error("type mismatch"):
        inst.ports.x.value = "hello"
    with diag_error("type mismatch"):
        inst.ports.n.value = 999

    # output ports cannot be driven from the python side
    with pytest.raises(hwl.VerilationException, match="Cannot set output port"):
        inst.ports.y.value = True

    # the correct way to set and read a port works
    inst.ports.x.value = True
    inst.step(1)
    assert inst.ports.y.value is True
    assert inst.ports["y"].value is True


def test_verilator_rebuild_same_dir(tmp_dir: Path):
    src_pass = "module top ports(x: in async bool, y: out async bool) { comb { y = x; } }"""
    src_inv = "module top ports(x: in async bool, y: out async bool) { comb { y = !x; } }"""

    inst_pass = compile_custom(src_pass).resolve("top.top").as_verilated(tmp_dir).instance()
    inst_inv = compile_custom(src_inv).resolve("top.top").as_verilated(tmp_dir).instance()

    def check(inst: hwl.VerilatedInstance, x: bool, y: bool):
        inst.ports.x.value = x
        inst.step(1)
        assert inst.ports.y.value is y

    check(inst_pass, False, False)
    check(inst_pass, True, True)
    check(inst_inv, False, True)
    check(inst_inv, True, False)
