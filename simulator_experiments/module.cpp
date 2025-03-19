#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "wrapper.cpp"

namespace py = pybind11;

class Ports {
public:
    Simulator *sim;
    Ports(Simulator &sim) : sim(&sim) {}
};

PYBIND11_MODULE(module, m) {
    py::class_<Ports>(m, "Ports")
        .def_property(
            "clk",
            [](Ports &ports) { return ports.sim->prev_ports.port_0_clk; },
            [](Ports &ports, bool val) { ports.sim->next_ports.port_0_clk = val; }
        )
        .def_property(
            "rst",
            [](Ports &ports) { return ports.sim->prev_ports.port_1_rst; },
            [](Ports &ports, bool val) { ports.sim->next_ports.port_1_rst = val; }
        )
        .def_property(
            "input_data",
            [](Ports &ports) { return ports.sim->prev_ports.port_2_input_data; },
            [](Ports &ports, std::array<bool, 16> val) { ports.sim->next_ports.port_2_input_data = val; }
        )
        .def_property(
            "input_valid",
            [](Ports &ports) { return ports.sim->prev_ports.port_3_input_valid; },
            [](Ports &ports, bool val) { ports.sim->next_ports.port_3_input_valid = val; }
        )
        .def_property_readonly(
            "input_ready",
            [](Ports &ports) { return ports.sim->prev_ports.port_4_input_ready; }
        )
        .def_property_readonly(
            "output_data",
            [](Ports &ports) { return ports.sim->prev_ports.port_5_output_data; }
        )
        .def_property_readonly(
            "output_valid",
            [](Ports &ports) { return ports.sim->prev_ports.port_6_output_valid; }
        )
        .def_property(
            "output_ready",
            [](Ports &ports) { return ports.sim->prev_ports.port_7_output_ready; },
            [](Ports &ports, bool val) { ports.sim->next_ports.port_7_output_ready = val; }
        );

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("step", &Simulator::step)
        .def_property_readonly(
            "ports",
            [](Simulator &sim) { return Ports(sim); },
            pybind11::return_value_policy::reference_internal
        );
}
