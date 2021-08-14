`include "scale_gen.sv"
`include "offset_gen.sv"

module pe_r_4 #(
    parameter INT_BW = 5,
    parameter FRA_BW = 7,
    parameter MUL_BW = 16,
    parameter ACC_BW = 32
)
(
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [1 : 0] gemm_uno, // 00: gemm; 01: div; 10: exp; 11: log
    input logic signed [ACC_BW-1 : 0] mac_i,
    input logic signed [MUL_BW-1 : 0] var_i,
    input logic signed [MUL_BW-1 : 0] x_i,
    input logic signed [MUL_BW-1 : 0] wc_i,
    input logic signed [ACC_BW-1 : 0] o_i,
    output logic signed [ACC_BW-1 : 0] mac_o,
    // output logic signed [MUL_BW-1 : 0] var_o,
    // output logic signed [MUL_BW-1 : 0] x_o,
    output logic signed [MUL_BW-1 : 0] wc_o,
    output logic signed [ACC_BW-1 : 0] o_o
);
    
    logic signed [MUL_BW-1 : 0] mac_t; // the left input of mul

    logic signed [MUL_BW-1 : 0] scale;
    logic signed [ACC_BW-1 : 0] offset;

    logic signed [INT_BW+FRA_BW : 0] mac_mux; // the left input of mul
    logic signed [INT_BW+FRA_BW : 0] var_mux; // the bottom input of mul
    logic signed [ACC_BW-1 : 0] acc_mux; // the bottom input of add

    logic signed [INT_BW+FRA_BW : 0] wreg;
    logic signed [MUL_BW-1 : 0] ireg;
    logic signed [MUL_BW-1 : 0] vreg;
    logic signed [ACC_BW-1 : 0] oreg;

    // mac
    always_comb begin : mac_i_trunc
        if (mac_i > {{(ACC_BW-INT_BW-FRA_BW*2){1'b0}}, {INT_BW{1'b1}}, {(FRA_BW*2){1'b1}}}) begin
            mac_t = {1'b0, {INT_BW{1'b1}}, {FRA_BW{1'b1}}};
        end
        else if (mac_i < {{(ACC_BW-INT_BW-FRA_BW*2){1'b1}}, {INT_BW{1'b0}}, {(FRA_BW*2){1'b0}}}) begin
            mac_t = {1'b1, {INT_BW{1'b0}}, {FRA_BW{1'b0}}};
        end else begin
            mac_t = mac_i[ACC_BW-1 : ACC_BW-MUL_BW];
        end
    end

    assign mac_mux = (gemm_uno == 2'b0) ? wreg : mac_t;
    assign var_mux = (gemm_uno == 2'b0) ? ireg[MUL_BW-1 : MUL_BW-1-INT_BW-FRA_BW] : scale[MUL_BW-1 : MUL_BW-1-INT_BW-FRA_BW];
    assign acc_mux = (gemm_uno == 2'b0) ?  o_i : offset;

    always_ff @(posedge clk or negedge rst_n) begin : mac_output
        if(~rst_n) begin
            oreg <= 0;
        end else begin
            oreg <= mac_mux * var_mux + acc_mux;
        end
    end
    assign mac_o = oreg;
    assign o_o = oreg;

    // wreg
    always_ff @(posedge clk or negedge rst_n) begin : wreg_output
        if (~rst_n) begin
            wreg <= 0;
        end else begin
            wreg <= wc_i;
        end
    end
    assign wc_o = wreg;

    // ireg
    always_ff @(posedge clk or negedge rst_n) begin : ireg_output
        if (~rst_n) begin
            ireg <= 0;
        end else begin
            ireg <= x_i;
        end
    end

    // scale
    scale_gen #(
        .INT_BW(INT_BW),
        .FRA_BW(FRA_BW),
        .MUL_BW(MUL_BW)
    ) U_scale_gen(
        .clk(clk),
        .rst_n(rst_n),
        .gemm_uno(gemm_uno),
        .x_i(x_i), 
        .y_i(wc_i), 
        .scale_o(scale));

    // offset
    offset_gen #(
        .INT_BW(INT_BW),
        .FRA_BW(FRA_BW),
        .MUL_BW(MUL_BW),
        .ACC_BW(ACC_BW)
    ) U_offset_gen(
        .clk(clk),
        .rst_n(rst_n),
        .gemm_uno(gemm_uno),
        .x_i(x_i), 
        .offset_o(offset));

endmodule