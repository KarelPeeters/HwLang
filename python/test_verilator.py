import os
import random
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

print("Creating simulator")
sim = top_verilated.instance(trace_path=build_dir + "/trace.vcd")
ports = sim.ports

for port_name in ports:
    port = ports[port_name]
    print(" ", port.name, port.direction, port.type)

print("Simulating")
random.seed(0xdeadbeef + 1)
input_length = 32
input_data = [random.randrange(2) != 0 for _ in range(input_length)]
input_width = 8

output_width = 8
output_data = []

# reset
ports.rst.value = True
ports.clk.value = False
sim.step(1)
ports.rst.value = False
ports.clk.value = True
sim.step(1)
ports.clk.value = False
sim.step(1)

input_data_left = list(input_data)

for i in range(16):
    print(f"  clock cycle {i}")
    # print(f"  ports: {({p: ports[p].value for p in ports})}")

    # [posedge]
    # send input
    # TODO all writes should actually happen after the clock edge, while all reads should happen before
    next_input_valid = ports.input_valid.value
    next_input_data = ports.input_data.value
    next_output_ready = ports.output_ready.value

    if i >= 1:
        if next_input_valid and ports.input_ready.value:
            print(f"  PY input handshake")
            next_input_valid = False
            next_input_data = [False] * input_width
        if not next_input_valid and len(input_data_left) > 0:
            next_input_valid = True
            next_input_data = input_data_left[:input_width]
            input_data_left = input_data_left[input_width:]

    # read output
    if i >= 3:
        if ports.output_valid.value and next_output_ready:
            print(f"  PY output handshake")
            output_data.extend(ports.output_data.value)
        next_output_ready = True

    # set clock
    ports.clk.value = True
    sim.step(0)

    # actually do writes
    ports.input_valid.value = next_input_valid
    ports.input_data.value = next_input_data
    ports.output_ready.value = next_output_ready
    sim.step(1)

    # [negedge]
    # set clock
    ports.clk.value = False
    sim.step(1)

sim.save_trace()

print("Input data:  ", "".join(str(int(x)) for x in input_data))
print("Output data: ", "".join(str(int(x)) for x in output_data))

# plt.plot(input_data, label="output")
# plt.plot(output_data, label="output")
# plt.legend()
# plt.show()
