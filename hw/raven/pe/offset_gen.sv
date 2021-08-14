`include "priority_enc_16.sv"

module offset_gen #(
    parameter INT_BW = 5,
    parameter FRA_BW = 10,
    parameter MUL_BW = 16,
    parameter ACC_BW = 32
)
(
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [1 : 0] gemm_uno, // 00: gemm; 01: div; 10: exp; 11: log
    input logic signed [MUL_BW-1 : 0] x_i,
    output logic signed [ACC_BW-1 : 0] offset_o
);
    
    logic [4 : 0] shiftx;
    logic [ACC_BW-1 : 0] log_offset;
    logic [ACC_BW-1 : 0] offset_x;

    priority_enc_16 U_priority_enc_16(.in(x_i), .out(shiftx));

    assign offset_x =   (gemm_uno == 2'b01) ? {MUL_BW{1'b0}} : // div: 0
                        (gemm_uno == 2'b10) ? {MUL_BW{1'b0}} : // exp: 0
                        (gemm_uno == 2'b11) ? log_offset : {MUL_BW{1'b0}}; // log: log(2^(ex))

    // offset
    always_ff @(posedge clk or negedge rst_n) begin : offset_output
        if (~rst_n) begin
            offset_o <= 0;
        end else begin
            offset_o <= offset_x;
        end
    end

    always_ff @(*) begin : proc_log_offset
        case (shiftx)
            5'b00000 : log_offset <= 'b0001000000000000;
            5'b00001 : log_offset <= 'b0010101110000000;
            5'b00010 : log_offset <= 'b0111011001000000;
            5'b00011 : log_offset <= 'b0111111111110000;
            5'b00100 : log_offset <= 'b0111111111110000;
            5'b00101 : log_offset <= 'b0111111111110000;
            5'b00110 : log_offset <= 'b0111111111110000;
            5'b00111 : log_offset <= 'b0111111111110000;
            5'b01000 : log_offset <= 'b0000000101100000;
            5'b01001 : log_offset <= 'b1010000000111110;
            5'b01010 : log_offset <= 'b0010000101000100;
            5'b01011 : log_offset <= 'b1110001101110100;
            5'b01100 : log_offset <= 'b0000100101100000;
            5'b01101 : log_offset <= 'b1011001011111100;
            5'b01110 : log_offset <= 'b0111111111111100;
            5'b01111 : log_offset <= 'b1111111111111000;
            5'b10000 : log_offset <= 'b0001000000000000;
            5'b10001 : log_offset <= 'b0010101110000000;
            5'b10010 : log_offset <= 'b0111011001001000;
            5'b10011 : log_offset <= 'b0111111111110010;
            5'b10100 : log_offset <= 'b0111111111010000;
            5'b10101 : log_offset <= 'b0111110011110000;
            5'b10110 : log_offset <= 'b0111111111110000;
            5'b10111 : log_offset <= 'b0111111111110000;
            5'b11000 : log_offset <= 'b0000000101100000;
            5'b11001 : log_offset <= 'b1010000000111110;
            5'b11010 : log_offset <= 'b0010000101000100;
            5'b11011 : log_offset <= 'b1110001101110100;
            5'b11100 : log_offset <= 'b0000100101100000;
            5'b11101 : log_offset <= 'b1011001011111100;
            5'b11110 : log_offset <= 'b0111111111111100;
            5'b11111 : log_offset <= 'b1111111111111000;
            default : log_offset <= 'b0000000000000000;
        endcase
    end

endmodule