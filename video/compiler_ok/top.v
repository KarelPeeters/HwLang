// module top
//   defined in "top"
module top(
    input clk, // clock bool
    input rst, // async bool
    input [2:0] data_to_add, // sync(clk, rst) int(0..8)
    output reg stream_in_ready, // sync(clk, rst) bool
    input stream_in_valid, // sync(clk, rst) bool
    input [2:0] stream_in_data, // sync(clk, rst) int(0..8)
    input stream_out_ready, // sync(clk, rst) bool
    output reg stream_out_valid, // sync(clk, rst) bool
    output reg [3:0] stream_out_data, // sync(clk, rst) int(0..16)
    output reg [31:0] values_transferred // sync(clk, rst) int(0..4294967296)
);
    wire dummy = 1'b0;

    always @(*) begin: comb_0
        reg dummy_var;
        reg stream_out_ready_0;
        reg stream_in_valid_0;
        reg [2:0] stream_in_data_0;
        reg [2:0] data_to_add_0;
        reg [3:0] tmp_0;

        dummy_var = dummy;
        stream_out_ready_0 = 1'bx;
        stream_in_valid_0 = 1'bx;
        stream_in_data_0 = 3'bx;
        data_to_add_0 = 3'bx;
        tmp_0 = 4'bx;

        stream_out_ready_0 = stream_out_ready;
        stream_in_ready = stream_out_ready_0;
        stream_in_valid_0 = stream_in_valid;
        stream_out_valid = stream_in_valid_0;
        stream_in_data_0 = stream_in_data;
        data_to_add_0 = data_to_add;
        tmp_0 = (4'd0 + stream_in_data_0) + (4'd0 + data_to_add_0);
        stream_out_data = tmp_0;
    end

    always @(posedge clk, posedge rst) begin: clocked_0
        reg stream_out_valid_0;
        reg stream_out_ready_0;
        reg [31:0] values_transferred_0;
        reg [31:0] shadow_values_transferred;
        reg [32:0] tmp_0;
        reg [33:0] tmp_1;
        reg [31:0] tmp_2;

        stream_out_valid_0 = 1'bx;
        stream_out_ready_0 = 1'bx;
        values_transferred_0 = 32'bx;
        tmp_0 = 33'bx;
        tmp_1 = 34'bx;
        tmp_2 = 32'bx;

        if (rst) begin
            values_transferred <= 32'd0;
        end else begin
            shadow_values_transferred = values_transferred;

            stream_out_valid_0 = stream_out_valid;
            stream_out_ready_0 = stream_out_ready;
            if ((stream_out_valid_0 & stream_out_ready_0)) begin
                values_transferred_0 = shadow_values_transferred;
                tmp_0 = (33'd0 + values_transferred_0) + 33'd1;
                tmp_1 = $signed((34'd0 + tmp_0)) % 34'sd4294967296;
                tmp_2 = tmp_1;
                values_transferred <= tmp_2;
                shadow_values_transferred = tmp_2;
            end
        end
    end
endmodule
