module adder_tree_64 #(
    parameter ACC_BW = 32
)(
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [ACC_BW-1 : 0] in [63 : 0],
    output logic [ACC_BW-1 : 0] out_l1 [15 : 0],
    output logic [ACC_BW-1 : 0] out_l2 [ 3 : 0],
    output logic [ACC_BW-1 : 0] out_l3
);
    logic [ACC_BW+1 : 0] sum_l1 [15 : 0];
    logic [ACC_BW+3 : 0] sum_l2 [ 3 : 0];
    logic [ACC_BW+5 : 0] sum_l3;

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin
            assign out_l1[i] = sum_l1[i][ACC_BW+1 : 2];
            always_ff @(posedge clk or negedge rst_n) begin : proc_sum_l1
                if(~rst_n) begin
                    sum_l1[i] <= 0;
                end else begin
                    sum_l1[i] <= in[i*4 + 3] + in[i*4 + 2] + in[i*4 + 1] + in[i*4 + 0];
                end
            end
        end
    endgenerate

    genvar i;
    generate
        for (i = 0; i < 4; i++) begin
            assign out_l2[i] = sum_l2[i][ACC_BW+3 : 4];
            always_ff @(posedge clk or negedge rst_n) begin : proc_sum_l2
                if(~rst_n) begin
                    sum_l2[i] <= 0;
                end else begin
                    sum_l2[i] <= sum_l1[i*4 + 3] + sum_l1[i*4 + 2] + sum_l1[i*4 + 1] + sum_l1[i*4 + 0];
                end
            end
        end
    endgenerate

    assign out_l3 = sum_l3[ACC_BW+5 : 6];
    always_ff @(posedge clk or negedge rst_n) begin : proc_sum_l3
        if(~rst_n) begin
            sum_l3 <= 0;
        end else begin
            sum_l3 <= sum_l2[ 3] + sum_l2[ 2] + sum_l2[ 1] + sum_l2[ 0];
        end
    end

endmodule