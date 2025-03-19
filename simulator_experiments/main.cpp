#include "wrapper.cpp"

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