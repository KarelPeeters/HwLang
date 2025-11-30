from hwl_sandbox.common.util import compile_custom


# TODO also test hardware prints
def test_print_compile():
    c = compile_custom("""
        import std.util.print;
        import std.types.[int, str];
        
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
