#include <cstdint>
#include <iostream>
#include <memory>

#include "verilated.h"
#include "verilated_vcd_c.h"
#include "/*[TEMPLATE-TOP-CLASS-NAME]*/.h"

typedef uint8_t Result;
const Result SUCCESS = 0;
const Result FAIL_CHECK = 1;
const Result FAIL_WRONG_LEN = 2;
const Result FAIL_DATA_NULL = 3;
const Result FAIL_INVALID_PORT = 4;
const Result FINISH = 16;

// Port get/set glue functions.
template <typename T>
Result get_port_primitive(T &port, size_t data_len, uint8_t *data) {
    size_t expected_len = sizeof(port);
    if (data_len != expected_len) return FAIL_WRONG_LEN;
    if (!data) return FAIL_DATA_NULL;

    *data = port;
    return SUCCESS;
}

Result get_port_impl(uint8_t const &port, size_t data_len, uint8_t *data) {
    return get_port_primitive(port, data_len, data);
}
Result get_port_impl(uint16_t const &port, size_t data_len, uint8_t *data) {
    return get_port_primitive(port, data_len, data);
}
Result get_port_impl(uint32_t const &port, size_t data_len, uint8_t *data) {
    return get_port_primitive(port, data_len, data);
}
Result get_port_impl(uint64_t const &port, size_t data_len, uint8_t *data) {
    return get_port_primitive(port, data_len, data);
}
template <size_t W>
Result get_port_impl(VlWide<W> &port, size_t data_len, uint8_t *data) {
    size_t expected_len = W * sizeof(EData);
    if (data_len != expected_len) return FAIL_WRONG_LEN;
    if (!data) return FAIL_DATA_NULL;

    for (size_t i = 0; i < W; ++i) {
        // Manually convert bytes to words to avoid strict aliasing issues.
        EData word = port.at(i);
        for (size_t j = 0; j < sizeof(EData); ++j) {
            data[i * sizeof(EData) + j] = (word >> (j * 8)) & 0xFF;
        }
    }
    return SUCCESS;
}

template <typename T>
Result set_port_primitive(T &port, size_t data_len, uint8_t const *data) {
    size_t expected_len = sizeof(port);
    if (data_len != expected_len) return FAIL_WRONG_LEN;
    if (!data) return FAIL_DATA_NULL;

    port = *data;
    return SUCCESS;
}

Result set_port_impl(uint8_t &port, size_t data_len, uint8_t const *data) {
    return set_port_primitive(port, data_len, data);
}
Result set_port_impl(uint16_t &port, size_t data_len, uint8_t const *data) {
    return set_port_primitive(port, data_len, data);
}
Result set_port_impl(uint32_t &port, size_t data_len, uint8_t const *data) {
    return set_port_primitive(port, data_len, data);
}
Result set_port_impl(uint64_t &port, size_t data_len, uint8_t const *data) {
    return set_port_primitive(port, data_len, data);
}
template <size_t W>
Result set_port_impl(VlWide<W> &port, size_t data_len, uint8_t const *data) {
    size_t expected_len = W * sizeof(EData);
    if (data_len != expected_len) return FAIL_WRONG_LEN;
    if (!data) return FAIL_DATA_NULL;

    for (size_t i = 0; i < W; ++i) {
        // Manually convert bytes to words to avoid strict aliasing issues.
        EData word = 0;
        for (size_t j = 0; j < sizeof(EData); ++j) {
            word |= (data[i * sizeof(EData) + j] << (j * 8));
        }
        port.data()[i] = word;
    }
    return SUCCESS;
}

// TODO rename to instance
// The wrapper instance returned and operated on by the C API.
class Wrapper {
   public:
    VerilatedContext *context;
    /*[TEMPLATE-TOP-CLASS-NAME]*/ *top;
    VerilatedVcdC *trace;

    Wrapper(char *trace_path) : context(new VerilatedContext()), top(new /*[TEMPLATE-TOP-CLASS-NAME]*/(context)), trace(nullptr) {
        if (trace_path) {
            context->traceEverOn(true);
            trace = new VerilatedVcdC();
            top->trace(trace, 1024);
            trace->open(trace_path);
        }
    }

    ~Wrapper() {
        delete trace;
        delete top;
        delete context;
    }
};

// TODO for verilator, generate an extra wrapper layer
//     that prefixes all ports so they never conflict with any builtin
//     properties
extern "C" {
    Result check(uint16_t arena_random, size_t top_module_index) {
        const uint16_t expected_arena_random = /*[TEMPLATE-ARENA-RANDOM]*/;
        const size_t expected_top_module_index = /*[TEMPLATE-TOP-MODULE-INDEX]*/;
        if (arena_random != expected_arena_random || top_module_index != expected_top_module_index) {
            return FAIL_CHECK;
        }
        return SUCCESS;
    }

    Wrapper *create_instance(char *trace_path) {
        return new Wrapper(trace_path);
    }

    void destroy_instance(Wrapper *wrapper) {
        // TODO final code somewhere?
        // wrapper->top.final();
        // wrapper->context.statsPrintSummary();

        delete wrapper;
    }

    Result step(Wrapper *wrapper, uint64_t increment_time) {
        if (wrapper->context->gotFinish()) {
            return FINISH;
        }

        wrapper->context->timeInc(increment_time);
        wrapper->top->eval();

        if (wrapper->trace) {
            wrapper->trace->dump(wrapper->context->time());
        }

        return SUCCESS;
    }

    void save_trace(Wrapper *wrapper) {
        if (wrapper->trace) {
            wrapper->trace->close();
            delete wrapper->trace;
            wrapper->trace = nullptr;
        }
    }

    Result get_port(Wrapper *wrapper, uint32_t port_index, size_t data_len, uint8_t *data) {
        // TODO generate this
        // clang-format off
        switch (port_index) {
            /*[TEMPLATE-PORTS-GET]*/
            default: return FAIL_INVALID_PORT;
        }
        // clang-format on
    }

    Result set_port(Wrapper *wrapper, uint32_t port_index, size_t data_len, uint8_t const *data) {
        // TODO generate this (only for input ports)
        // TODO check somewhere that sync signals only change on the right edges
        // clang-format off
        switch (port_index) {
            /*[TEMPLATE-PORTS-SET]*/
            default: return FAIL_INVALID_PORT;
        }
        // clang-format on
    }
}
