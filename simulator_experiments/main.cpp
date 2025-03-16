#include "simulator.cpp"
#include <cstdio>
#include <utility>

const uint32_t FIXPOINT_ITERATIONS = 4;

typedef ModuleSignals_0 TopSignals;
typedef ModulePortsVal_0 TopPortsVal;

class Simulator {
public:
    TopSignals prev_signals;
    TopPortsVal prev_ports;
    TopSignals next_signals;
    TopPortsVal next_ports;
    Simulator() {
        // TODO initialize everything as undefined once that concept exists
        prev_signals = {};
        prev_ports = {};
        next_signals = {};
        next_ports = {};
    };

    void step() {
        // run until (hopefully) fixpoint
        for (uint32_t i = 0; i < FIXPOINT_ITERATIONS; i++) {
            module_0_all(prev_signals, prev_ports.as_ptrs(), next_signals, next_ports.as_ptrs());
        }
        // everything will stay the same by default
        // TODO can we implement this as a pointer swap, or do we actually need to do a copy?
        prev_signals = next_signals;
        prev_ports = next_ports;
    }

    void port_set_clk(bool clk) {
        next_ports.port_0_clk = clk;
    }
};

void print_state(Simulator &sim) {
    // printf("clk=%d, rst=%d\n", sim.prev_ports.port_0_clk, sim.prev_ports.port_1_rst);
    // // printf("level: %ld\n", sim.prev_signals.child_0.reg_1_level);
    // printf("input: valid=%d, ready=%d, data=%d\n", sim.prev_ports.port_3_input_valid, sim.prev_ports.port_4_input_ready, sim.prev_ports.port_2_input_data[0]);
    // printf("output: valid=%d, ready=%d, data=%d\n", sim.prev_ports.port_6_output_valid, sim.prev_ports.port_7_output_ready, sim.prev_ports.port_5_output_data[0]);
    // printf("\n");
}

void step(Simulator &sim) {
    printf("======= step =======\n");
    sim.step();
    print_state(sim);
}

int main() {
    auto sim = Simulator();
    step(sim);
    sim.next_ports.port_1_rst = true;
    step(sim);
    sim.next_ports.port_1_rst = false;
    step(sim);
    sim.next_ports.port_0_clk = true;
    sim.next_ports.port_3_input_valid = true;
    sim.next_ports.port_2_input_data = {1, 1, 1, 1, 1, 1, 1, 1};
    step(sim);
    sim.next_ports.port_0_clk = false;
    step(sim);
    sim.next_ports.port_0_clk = true;
    sim.next_ports.port_3_input_valid = false;
    sim.next_ports.port_2_input_data = {};
    step(sim);
    sim.next_ports.port_0_clk = false;
    step(sim);
    sim.next_ports.port_0_clk = true;
    sim.next_ports.port_7_output_ready = 1;
    step(sim);
    sim.next_ports.port_0_clk = false;
    step(sim);
    sim.next_ports.port_0_clk = true;
    step(sim);
    sim.next_ports.port_0_clk = false;
    step(sim);
    sim.next_ports.port_0_clk = true;
    step(sim);
    sim.next_ports.port_0_clk = false;
    step(sim);
}