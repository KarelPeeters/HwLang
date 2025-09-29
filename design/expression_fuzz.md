TODO set up test framework for integer expressions in verilog

* driven by python
* generate some random integer ranges and a web of expressions that use them (including constants)
    * here we need to ask the compiler what the output range of expressions is to define good output port types
* generate corresponding and function and module wrapper
* call function and module with random args from python, compare results
    * (this will involve generated verilog, verilator testbench, ...)
* start with just integer expressions, later include booleans, arrays, structs, tuples
