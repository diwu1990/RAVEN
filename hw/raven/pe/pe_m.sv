module pe_m #(
    parameter INT_BW = 5,
    parameter FRA_BW = 10,
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
    output logic signed [MUL_BW-1 : 0] var_o,
    output logic signed [MUL_BW-1 : 0] x_o,
    output logic signed [MUL_BW-1 : 0] wc_o,
    output logic signed [ACC_BW-1 : 0] o_o
);
    
    logic signed [MUL_BW-1 : 0] mac_t; // the left input of mul

    logic signed [MUL_BW-1 : 0] mac_mux; // the left input of mul
    logic signed [MUL_BW-1 : 0] var_mux; // the bottom input of mul
    logic signed [ACC_BW-1 : 0] acc_mux; // the bottom input of add

    logic signed [MUL_BW-1 : 0] wreg;
    logic signed [MUL_BW-1 : 0] ireg;
    logic signed [MUL_BW-1 : 0] vreg;
    logic signed [ACC_BW-1 : 0] oreg;

    // mac
    always_comb begin : mac_i_trunc
        if (mac_i > {{(ACC_BW-INT_BW-FRA_BW*2){1'b0}}, {INT_BW{1'b1}}, {(FRA_BW*2){1'b1}}}) begin
            mac_t = {1'b0, {INT_BW{1'b1}}, {FRA_BW{1'b1}}}
        end
        else if (mac_i < {(ACC_BW-INT_BW-FRA_BW*2), {INT_BW{1'b0}}, {(FRA_BW*2)FRA_BW{1'b0}}}) begin
            mac_t = {1'b1, {INT_BW{1'b0}}, {FRA_BW{1'b0}}}
        end else begin
            mac_t = mac_i[ACC_BW-1 : ACC_BW-MUL_BW]
        end
    end

    assign mac_mux = (gemm_uno == 2'b0) ? wreg : mac_t;
    assign var_mux = (gemm_uno == 2'b0) ? ireg : vreg;
    assign acc_mux = (gemm_uno == 2'b0) ?  o_i : wreg;

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
    assign x_o = ireg;

    // vreg
    always_ff @(posedge clk or negedge rst_n) begin : vreg_output
        if (~rst_n) begin
            vreg <= 0;
        end else begin
            vreg <= var_i;
        end
    end
    assign var_o = vreg;

endmodule