import module
from vcd import VCDWriter

sim = module.Simulator()

# TODO actually plot ints
labels = ['clk', 'rst', 'input_data', 'input_valid', 'input_ready', 'output_data', 'output_valid', 'output_ready']
data = []

def step():
    sim.step()
    data.append((
        sim.port_clk, sim.port_rst,
        sim.port_input_data[0], sim.port_input_valid, sim.port_input_ready,
        sim.port_output_data[0], sim.port_output_valid, sim.port_output_ready,
    ))

def plot():
    with VCDWriter(open('wave.vcd', 'w')) as writer:
        vars = [writer.register_var('top', label, 'wire', 1) for label in labels]
        for t, line in enumerate(data):
            for i, var in enumerate(vars):
                writer.change(var, t, line[i])
            

def cycle():
    sim.port_clk = True
    step()
    sim.port_clk = False
    step()

step()
sim.port_rst = True
cycle()
sim.port_rst = False
cycle()
sim.port_input_valid = True
sim.port_input_data = [True] * 16
cycle()
sim.port_input_valid = False
cycle()
cycle()
cycle()
sim.port_output_ready = True
cycle()
sim.port_output_ready = False
cycle()
cycle()
sim.port_output_ready = True
cycle()
sim.port_output_ready = False
cycle()

plot()
