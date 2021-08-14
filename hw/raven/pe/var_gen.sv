`include "priority_enc_16.sv"

module var_gen #(
    parameter INT_BW = 5,
    parameter FRA_BW = 10,
    parameter MUL_BW = 16
)
(
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [1 : 0] gemm_uno, // 00: gemm; 01: div; 10: exp; 11: log
    input logic signed [MUL_BW-1 : 0] x_i,
    output logic signed [MUL_BW-1 : 0] var_o
);
    
    logic [MUL_BW-1 : 0] x_norm;
    logic [INT_BW-1 : 0] x_int;
    logic [FRA_BW-1 : 0] x_frac;
    logic [4 : 0] shiftx;
    logic [MUL_BW-1 : 0] var_x;
    logic [MUL_BW-1 : 0] point_sub_x;

    assign x_int = x_i[MUL_BW-1 : FRA_BW];
    assign x_frac = x_i[FRA_BW-1 : 0];

    priority_enc_16 U_priority_enc_16(.in(x_i), .out(shiftx));
    assign x_norm = x_i << shiftx;

    assign point_sub_x = {1'b0, {INT_BW{1'b0}}, 2'b11, {(FRA_BW-2){1'b0}}} - x_norm;

    assign var_x  = (gemm_uno == 2'b01) ? point_sub_x : // div: POINT - x_norm
                    (gemm_uno == 2'b10) ? {{(INT_BW+1){x_i[MUL_BW-1]}}, x_frac} : // exp: x_frac
                    (gemm_uno == 2'b11) ? point_sub_x : {MUL_BW{1'b0}} ; // log: POINT - x_norm

    // var
    always_ff @(posedge clk or negedge rst_n) begin : vreg_output
        if (~rst_n) begin
            var_o <= 0;
        end else begin
            var_o <= var_x;
        end
    end

endmodule