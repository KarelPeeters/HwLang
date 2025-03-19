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
};
