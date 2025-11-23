# from hwl_sandbox.common.util import compile_custom
#
#
# def test_print_const_literals():
#     c = compile_custom("""
#         import std.util.print;
#         fn f() {
#             print("test");
#             print(8);
#             print(false);
#             print([1, 2, 3]);
#         }
#     """)
#     f = c.resolve("top.f")
#     with c.capture_prints() as capture:
#         f()
#     assert capture.prints == [
#         "test",
#         "8",
#         "false",
#         "[1, 2, 3]",
#     ]
#
#
# # def test_print_():
# #     c = compile_custom("""import std.util.print; fn f() { print("test"); }""")
# #     f = c.resolve("top.f")
# #     with c.capture_prints() as capture:
# #         f()
# #     assert capture.prints == ["test"]
#
# # TODO test mixed-compile-hardware structs and tuples
# # TODO significantly expand expression testing, in particular type expansion and int/bits conversions
#
