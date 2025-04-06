from pathlib import Path

import hwl

# TODO add this to CI
source = hwl.Source(str(Path(__file__).parent / "../../design/project"))
print("Parsed files:")
for f in source.files:
    print(f"  {f}")

parsed = source.parse()
print(parsed)

compile = parsed.compile()

top = compile.resolve("top.top")
foo_function = compile.resolve("top.foo_function")
foo_module = compile.resolve("top.foo_module")
print(top, foo_function, foo_module)

# TODO call function
print(foo_function(4, 5))
print(foo_function(4, b=5))

# TODO instantiate module into generated verilog
ty_bool = compile.resolve("std.types.bool")
foo_inst = foo_module.instance(T=ty_bool)

foo_verilog = foo_inst.generate_verilog()
print(foo_verilog.source)

# TODO instantiate module into simulator
# foo_sim = foo_inst.to_simulator()
