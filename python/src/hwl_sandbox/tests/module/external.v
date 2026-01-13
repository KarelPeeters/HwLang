module external_module#(
    parameter W = 8
)(
    input [W-1:0] x,
    output [W:0] y
);
    assign y = x + 1;
endmodule
