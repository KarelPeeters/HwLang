#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "wrapper.cpp"

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(module, m) {
    // m.def("add", &add, "A function which adds two numbers");
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("step", &Simulator::step)
        .def_property(
            "port_clk",
            [](Simulator &sim) { return sim.prev_ports.port_0_clk; },
            [](Simulator &sim, bool val) { sim.next_ports.port_0_clk = val; }
        )
        .def_property(
            "port_rst",
            [](Simulator &sim) { return sim.prev_ports.port_1_rst; },
            [](Simulator &sim, bool val) { sim.next_ports.port_1_rst = val; }
        )
        .def_property(
            "port_input_data",
            [](Simulator &sim) { return sim.prev_ports.port_2_input_data; },
            [](Simulator &sim, std::array<bool, 16> val) { sim.next_ports.port_2_input_data = val; }
        )
        .def_property(
            "port_input_valid",
            [](Simulator &sim) { return sim.prev_ports.port_3_input_valid; },
            [](Simulator &sim, bool val) { sim.next_ports.port_3_input_valid = val; }
        )
        .def_property_readonly(
            "port_input_ready",
            [](Simulator &sim) { return sim.prev_ports.port_4_input_ready; }
        )
        .def_property_readonly(
            "port_output_data",
            [](Simulator &sim) { return sim.prev_ports.port_5_output_data; }
        )
        .def_property_readonly(
            "port_output_valid",
            [](Simulator &sim) { return sim.prev_ports.port_6_output_valid; }
        )
        .def_property(
            "port_output_ready",
            [](Simulator &sim) { return sim.prev_ports.port_7_output_ready; },
            [](Simulator &sim, bool val) { sim.next_ports.port_7_output_ready = val; }
        );
}
