from pathlib import Path

import hwl

from hwl_sandbox.common.util import compile_custom


def test_print_compile():
    c = compile_custom("""
        struct Point(T: type) { x: T, y: T }
        enum Option(T: type) { None, Some(T) }
        
        fn f() {
            print("test");
            print(8);
            print(false);
            
            print([1, 2, 3]);
            
            print(());
            print((1,));
            print((1, 2, 3));
            
            print(Point(int).new(x=4, y=8));
            print(Option(int).None);
            print(Option(int).Some(10));
        }
    """)
    expected_prints = [
        "test\n",
        "8\n",
        "false\n",
        "[1, 2, 3]\n",
        "()\n",
        "(1,)\n",
        "(1, 2, 3)\n",
        "Point(T=int).new(x=4, y=8)\n",
        "Option(T=int).None\n",
        "Option(T=int).Some(10)\n",
    ]

    f = c.resolve("top.f")
    with c.capture_prints() as capture:
        f()
    assert capture.prints == expected_prints


# TODO capture outputs and check that they are correct
def test_print_hardware():
    c = compile_custom("""
        struct Point(T: type) { x: T, y: T }
        enum Option(T: type) { None, Some(T) }
        
        module f ports(x: in async uint(4)) {
            comb {
                print("{x == 0}");
                print("{x}");
                print("{(x)}");
                print("{(x, x)}");
                print("{[x, x]}");
                print("{Point(uint(4)).new(x=x, y=x)}");
                print("{Option(uint(4)).None}");
                print("{Option(uint(4)).Some(x)}");
            }
        }
    """)
    f: hwl.Module = c.resolve("top.f")
    print(f.as_verilog())


def test_print_hardware_array_length_using_all_bits(tmp_dir: Path):
    # Check that the generated IR for loop does not become an infinite loop
    src = """
        module f ports(clk: in clock, x: in sync(clk) [4]bool) {
        clocked(clk) {
            print(x);
        }
    }
    """
    c = compile_custom(src)
    f: hwl.Module = c.resolve("top.f")
    print(f.as_verilog())
    inst = f.as_verilated(tmp_dir).instance()

    inst.ports.x.value = [False, True, False, True]
    inst.ports.clk.value = False
    inst.step(1)
    inst.ports.clk.value = True
    inst.step(1)
