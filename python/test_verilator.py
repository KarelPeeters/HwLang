import os
from pathlib import Path

import hwl

source = hwl.Source(str(Path(__file__).parent / "../design/project"))
parsed = source.parse()
compiled = parsed.compile()

print("Resolving")
top = compiled.resolve("top.top")
print(top)

print("Generating verilog")
print(top.as_verilog())

print("Verilating")
build_dir = "./build"
os.makedirs(build_dir, exist_ok=True)
top_verilated = top.as_verilated(build_dir=build_dir)
print(top_verilated)

top_instance = top_verilated.instance()
print(top_instance)
print(top_instance.ports)
print(top_instance.ports.clk)
print(list(top_instance.ports))
print(top_instance.ports.clk.value)

top_instance.ports.clk.value = True
print(top_instance.ports.clk.value)
top_instance.ports.clk.value = False
print(top_instance.ports.clk.value)

for port_name in top_instance.ports:
    port = top_instance.ports[port_name]
    print(port.name, port.direction, port.type, port.value)

print(top_instance.ports.clk.value)

# test async propagation
top_instance.ports.huge_input.value = 0
top_instance.step(1)
print(top_instance.ports.huge_output_comb.value)
print(top_instance.ports.huge_output_clocked.value)

top_instance.ports.huge_input.value = 7
top_instance.step(1)

print("after comb")
print(top_instance.ports.huge_output_comb.value)
print(top_instance.ports.huge_output_clocked.value)

top_instance.ports.clk.value = True
top_instance.step(1)

print("after pos")
print(top_instance.ports.huge_output_comb.value)
print(top_instance.ports.huge_output_clocked.value)

top_instance.ports.clk.value = False
top_instance.step(1)

print("after neg")
print(top_instance.ports.huge_output_comb.value)
print(top_instance.ports.huge_output_clocked.value)
