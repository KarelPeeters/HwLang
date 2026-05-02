module external_module#(
    parameter W = 8
)(
    input [W-1:0] x,
    output [W:0] y
);
    assign y = x + 1;
endmodule

module external_module_zero(
    // omit x, zero width
    output [0:0] y
);
    assign y = 1;
endmodule

module external_module_no_ports();
endmodule

module \input ();
endmodule
