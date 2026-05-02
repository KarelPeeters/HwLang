from pathlib import Path

import hwl
import pytest

from hwl_sandbox.common.util import compile_custom


def test_port_reg_name(tmpdir: Path):
    src = """
    module foo ports(clk: in clock, a: out sync(clk) bool) {
        clocked(clk) {
            reg wire a = undef;
            a = !a;
        }
    }
    """

    c = compile_custom(src)
    m: hwl.Module = c.resolve("top.foo")
    m.as_verilated(tmpdir)


def test_module_instance_simple(tmp_dir: Path):
    src = """
    module parent ports(x: in async bool, y: out async bool) { instance child ports(x, y); }
    module child ports(x: in async bool, y: out async bool) { comb { y = x; } }
    """

    c = compile_custom(src)
    parent: hwl.Module = c.resolve("top.parent")
    print(parent.as_verilog().source)
    parent_verilated: hwl.ModuleVerilated = parent.as_verilated(tmp_dir)
    parent_inst: hwl.VerilatedInstance = parent_verilated.instance()

    for v in [False, True, False, True]:
        parent_inst.ports.x.value = v
        parent_inst.step(1)
        assert parent_inst.ports.y.value == v


def test_module_instance_generic(tmp_dir: Path):
    src = """
    module parent ports(x: in async uint(4), y: out async uint(4)) { instance child(N=4) ports(x, y); }
    module child(N: uint) ports(x: in async uint(N), y: out async uint(N)) { comb { y = x; } }
    """

    c = compile_custom(src)
    parent: hwl.Module = c.resolve("top.parent")
    print(parent.as_verilog().source)
    parent_verilated: hwl.ModuleVerilated = parent.as_verilated(tmp_dir)
    parent_inst: hwl.VerilatedInstance = parent_verilated.instance()

    for v in range(2 ** 4):
        parent_inst.ports.x.value = v
        parent_inst.step(1)
        assert parent_inst.ports.y.value == v


def test_wire_driven_by_child(tmp_dir: Path):
    src = """
    module parent ports(y: out const bool) {
        wire w: bool;
        instance child ports(y=w);
        comb { y = w; }
    }
    module child ports(y: out const bool) { comb { y = true; } }
    """

    c = compile_custom(src)
    parent: hwl.Module = c.resolve("top.parent")
    print(parent.as_verilog().source)
    parent_verilated: hwl.ModuleVerilated = parent.as_verilated(tmp_dir)
    parent_inst: hwl.VerilatedInstance = parent_verilated.instance()

    parent_inst.step(1)
    assert parent_inst.ports.y.value is True


def test_interface_access(tmp_dir: Path):
    src = """
    interface Foo {
        x: bool,
        interface input { x: in }
        interface output { x: out }
    }
    module top ports(a: interface async Foo.input, b: interface async Foo.output) {
        comb {
            b.x = a.x;
        }   
    }
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")
    print(top.as_verilog().source)
    top_verilated: hwl.ModuleVerilated = top.as_verilated(tmp_dir)
    top_inst: hwl.VerilatedInstance = top_verilated.instance()

    for v in [False, True, False, True]:
        top_inst.ports.a_x.value = v
        top_inst.step(1)
        assert top_inst.ports.b_x.value == v


def test_instantiate_external_module(tmp_dir: Path):
    src = """
    external module external_module(W: natural) ports(x: in async uint(W), y: out async uint(W+1))
    module top ports(x: in async uint(4), y: out async uint(5)) {
        instance external_module(W=4) ports(x, y);
    }
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")
    print(top.as_verilog().source)

    extra_verilog_files = [Path(__file__).parent / "external.v"]
    top_verilated: hwl.ModuleVerilated = top.as_verilated(tmp_dir, extra_verilog_files=extra_verilog_files)
    top_inst: hwl.VerilatedInstance = top_verilated.instance()

    for v in [0, 1, 2, 15]:
        top_inst.ports.x.value = v
        top_inst.step(1)
        assert top_inst.ports.y.value == v + 1


def test_instantiate_zero_width_ports(tmp_dir: Path):
    src = """
    module top ports(
        x: in async uint(0),
        y0: out async uint(0),
        y1: out async uint(0),
    ) {
        instance internal_module ports(x, y0=y0, y1=_);
        instance external_module_no_ports ports(x, y0=y1, y1=_);
    }

    module internal_module ports(
        x: in async uint(0),
        y0: out async uint(0),
        y1: out async uint(0),
    ) {
        comb {
            y0 = x;
            y1 = x;
        }
    }
    external module external_module_no_ports ports(
        x: in async uint(0),
        y0: out async uint(0),
        y1: out async uint(0),
    )
    
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")
    print(top.as_verilog().source)

    extra_verilog_files = [Path(__file__).parent / "external.v"]
    top_verilated: hwl.ModuleVerilated = top.as_verilated(tmp_dir, extra_verilog_files=extra_verilog_files)
    top_inst: hwl.VerilatedInstance = top_verilated.instance()

    top_inst.ports.x.value = 0
    top_inst.step(1)
    assert top_inst.ports.y0.value == 0
    assert top_inst.ports.y1.value == 0


def test_instantiate_external_module_no_ports(tmp_dir: Path):
    src = """
    external module external_module_no_ports ports()
    module top ports() {
        instance external_module_no_ports ports();
    }
    """
    c = compile_custom(src)
    top: hwl.Module = c.resolve("top.top")
    print(top.as_verilog().source)

    extra_verilog_files = [Path(__file__).parent / "external.v"]
    top_verilated: hwl.ModuleVerilated = top.as_verilated(tmp_dir, extra_verilog_files=extra_verilog_files)
    top_inst: hwl.VerilatedInstance = top_verilated.instance()

    top_inst.step(1)


def test_cyclic_instantiation_almost():
    src = """
    module foo(n: uint) ports(x: in async bool, y: out async bool) {
        if (n == 0) {
            comb { y = x; }
        } else {
            instance foo(n-1) ports(x, y);
        }
    }
    """
    c = compile_custom(src)
    foo: hwl.Module = c.resolve("top.foo")(4)
    print(foo.as_verilog().source)


def test_cyclic_instantiation_real():
    src = """
    module foo ports(x: in async bool, y: out async bool) { instance bar ports(x, y); }
    module bar ports(x: in async bool, y: out async bool) { instance foo ports(x, y); }
    """
    c = compile_custom(src)

    with pytest.raises(hwl.DiagnosticException, match="cyclic module instantiation"):
        foo: hwl.Module = c.resolve("top.foo")
        print(foo.as_verilog().source)


def test_undef_expression():
    src = """
    module top ports() {
        wire w: bool = undef;
        comb {
            val v: uint(8) = undef;
        }
    }
    """
    compile_custom(src).resolve("top.top")


def test_reference_escape_simple():
    src = """ 
    module top ports() {
        wire w: bool;
        instance child(ref(w)) ports();
    }
    module child(a: any) ports() {}
    """
    with pytest.raises(hwl.DiagnosticException, match="item parameters cannot contain references"):
        compile_custom(src).resolve("top.top")


def test_reference_escape_function():
    src = """
    module top ports() {
        wire w: bool = true;
        const r = ref(w);
        fn f() {
            deref(r).x = false;
        }
        instance child(f) ports();
    }
    module child(f: Function) ports() {
        comb { f(); }
    }
    """
    with pytest.raises(hwl.DiagnosticException, match="attempt to use signal reference outside its declaring module"):
        compile_custom(src).resolve("top.top")


def test_diamond_instantiation():
    src = """
    module a ports() { instance b ports(); instance c ports(); }
    module b ports() { instance c ports(); }
    module c ports() {}
    """
    m = compile_custom(src).resolve("top.a")
    print(m.as_verilog().source)


def test_interface_chain(tmp_dir: Path):
    src = """
    interface foo {
        d: uint(8),
        interface input { d: in }
        interface output { d: out }
    }
    module top ports(x: interface async foo.input, y: interface async foo.output) {
        wire w: interface foo;
        instance pass ports(x=x, y=w);  
        instance pass ports(x=w, y=y);  
    }
    module pass ports(x: interface async foo.input, y: interface async foo.output) {
        comb { y.d = x.d; } 
    }
    """
    m: hwl.Module = compile_custom(src).resolve("top.top")
    inst = m.as_verilated(tmp_dir).instance()

    for v in [0, 1, 2, 3]:
        inst.ports.x_d.value = v
        inst.step(1)
        assert inst.ports.y_d.value == v


def test_self_instance():
    src = """
    module top ports() {
        instance top ports();
    }
    """
    c = compile_custom(src)
    with pytest.raises(hwl.DiagnosticException, match="cyclic module instantiation"):
        c.resolve("top.top")


def test_module_inner_decls_can_access_outer():
    # Test that declarations inside the module can access values in parameters and even hardware signals.
    # This requires that flow/scope/capture handling is implemented at least somewhat correctly.

    src = """
    module foo(N: uint) ports() {
        wire w: bool = false;
    
        type array(T: type) = [N]T;
        type foo = type(w);
        
        wire w1: array(bool) = [false] * N;
        wire w2: foo = false;
    }
    """
    foo = compile_custom(src).resolve("top.foo")
    print(foo(N=4).as_verilog().source)
