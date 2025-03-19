import module
from vcd import VCDWriter

sim = module.Simulator()
ports = sim.ports

# TODO actually plot ints
labels = ['clk', 'rst', 'input_data', 'input_valid', 'input_ready', 'output_data', 'output_valid', 'output_ready']
data = []

def step():
    sim.step()
    data.append((
        ports.clk, ports.rst,
        ports.input_data[0], ports.input_valid, ports.input_ready,
        ports.output_data[0], ports.output_valid, ports.output_ready,
    ))

def plot():
    with VCDWriter(open('wave.vcd', 'w')) as writer:
        vars = [writer.register_var('top', label, 'wire', 1) for label in labels]
        for t, line in enumerate(data):
            for i, var in enumerate(vars):
                writer.change(var, t, line[i])
            

def cycle():
    ports.clk = True
    step()
    ports.clk = False
    step()

step()
ports.rst = True
cycle()
ports.rst = False
cycle()
ports.input_valid = True
ports.input_data = [True] * 16
cycle()
ports.input_valid = False
cycle()
cycle()
cycle()
ports.output_ready = True
cycle()
ports.output_ready = False
cycle()
cycle()
ports.output_ready = True
cycle()
ports.output_ready = False
cycle()

plot()
