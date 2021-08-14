`include "priority_enc_16.sv"

module scale_gen #(
    parameter INT_BW = 5,
    parameter FRA_BW = 10,
    parameter MUL_BW = 16
)
(
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [1 : 0] gemm_uno, // 00: gemm; 01: div; 10: exp; 11: log
    input logic signed [MUL_BW-1 : 0] x_i,
    input logic signed [MUL_BW-1 : 0] y_i,
    output logic signed [MUL_BW-1 : 0] scale_o
);
    
    logic [MUL_BW-1 : 0] x_norm;
    logic [INT_BW-1 : 0] x_int;
    logic [4 : 0] shiftx;
    logic [MUL_BW-1 : 0] exp_scale;
    logic [MUL_BW-1 : 0] scale_xy;

    assign x_int = x_i[MUL_BW-1 : FRA_BW];

    priority_enc_16 U_priority_enc_16(.in(x_i), .out(shiftx));

    // scale
    always_ff @(posedge clk or negedge rst_n) begin : scale_output
        if (~rst_n) begin
            scale_o <= 0;
        end else begin
            scale_o <= scale_xy;
        end
    end
    
    assign scale_xy =   (gemm_uno == 2'b01) ? y_i >> shiftx : //div; y*2^(-ex)
                        (gemm_uno == 2'b10) ? exp_scale : // exp: exp(x_int)
                        (gemm_uno == 2'b11) ? {MUL_BW{1'b1}} : {MUL_BW{1'b0}}; // log: -1

    always_ff @(*) begin : proc_exp_scale
        case (x_int)
            5'b00000 : exp_scale <= 16'b0001000000000000;
            5'b00001 : exp_scale <= 16'b0010101110000000;
            5'b00010 : exp_scale <= 16'b0111011001000000;
            5'b00011 : exp_scale <= 16'b0111111111110000;
            5'b00100 : exp_scale <= 16'b0111111111110000;
            5'b00101 : exp_scale <= 16'b0111111111110000;
            5'b00110 : exp_scale <= 16'b0111111111110000;
            5'b00111 : exp_scale <= 16'b0111111111110000;
            5'b01000 : exp_scale <= 16'b0000000101100000;
            5'b01001 : exp_scale <= 16'b1010000000111110;
            5'b01010 : exp_scale <= 16'b0010000101000100;
            5'b01011 : exp_scale <= 16'b1110001101110100;
            5'b01100 : exp_scale <= 16'b0000100101100000;
            5'b01101 : exp_scale <= 16'b1011001011111100;
            5'b01110 : exp_scale <= 16'b0111111111111100;
            5'b01111 : exp_scale <= 16'b1111111111111000;
            5'b10000 : exp_scale <= 16'b0001000000000000;
            5'b10001 : exp_scale <= 16'b0010101110000000;
            5'b10010 : exp_scale <= 16'b0111011001001000;
            5'b10011 : exp_scale <= 16'b0111111111110010;
            5'b10100 : exp_scale <= 16'b0111111111010000;
            5'b10101 : exp_scale <= 16'b0111110011110000;
            5'b10110 : exp_scale <= 16'b0111111111110000;
            5'b10111 : exp_scale <= 16'b0111111111110000;
            5'b11000 : exp_scale <= 16'b0000000101100000;
            5'b11001 : exp_scale <= 16'b1010000000111110;
            5'b11010 : exp_scale <= 16'b0010000101000100;
            5'b11011 : exp_scale <= 16'b1110001101110100;
            5'b11100 : exp_scale <= 16'b0000100101100000;
            5'b11101 : exp_scale <= 16'b1011001011111100;
            5'b11110 : exp_scale <= 16'b0111111111111100;
            5'b11111 : exp_scale <= 16'b1111111111111000;
            default : exp_scale <= 16'b0000000000000000;
        endcase
    end

endmodule