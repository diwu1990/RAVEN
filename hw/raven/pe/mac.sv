module mac #(
    parameter MUL_BW = 16,
    parameter ACC_BW = 32
)
(
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic signed [MUL_BW-1 : 0] A_i,
    input logic signed [MUL_BW-1 : 0] B_i,
    input logic signed [ACC_BW-1 : 0] C_i,
    output logic signed [ACC_BW-1 : 0] C_o
);
    
    always_ff @(posedge clk or negedge rst_n) begin : mac_output
        if(~rst_n) begin
            C_o <= 0;
        end else begin
            C_o <= A_i * B_i + C_i;
        end
    end

endmodule