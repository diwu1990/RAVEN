`include "param_def.sv"
`include "mac.sv"
`include "lut_exp_scale.sv"
`include "lut_log_offset.sv"

module uno (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input logic [1:0] op,
    input [`MAC_BW-1 : 0] X,
    input [`MAC_BW-1 : 0] Y,
    input [2*`MAC_BW-1 : 0] Z,
    input [`MAC_BW-1 : 0] coeff,
    input logic fisrt_cycle,
    input logic last_cycle,
    input logic acc_en
);

//MAC: op = 00
//div: op = 01
//exp: op = 10
//log: op = 11

logic [`MAC_BW-1 : 0] x_norm;
logic [3 : 0] x_int;
logic [`MAC_BW-1 : 0] exp_scale;
logic [`MAC_BW-1 : 0] point;
logic [3 : 0] shiftx;
logic [2*`MAC_BW-1 : 0] log_offset;

assign x_int = X[11 : 8];
assign exp_int = X[11 : 8];

assign x_norm = X << shiftx;

assign point_sub_x = point - x_norm;

lut_exp_scale U_lut_exp_scale(.exp_int(x_int), 
                              .exp_scale(exp_scale));

lut_log_offset U_lut_log_offset(.shift_offset(shiftx), 
                                .log_offset(log_offset));

priority_enc_12 U_priority_enc_12(.in(X),
                                  .out(shiftx));

assign scale  = (op == 2'b01) ? Y >> shiftx : //div; y*2^(-ex)
                (op == 2'b10) ? exp_scale : // exp: exp(x_int)
                (op == 2'b11) ? {`MAC_BW{1'b1}} : {`MAC_BW{1'b0}}; // log: -1

assign offset = (op == 2'b01) ? {`MAC_BW{1'b0}} : // div: 0
                (op == 2'b10) ? {`MAC_BW{1'b0}} : // exp: 0
                (op == 2'b11) ? log_offset : {`MAC_BW{1'b0}}; // log: log(2^(ex))

assign point  = (op == 2'b01) ? 12'b000011000000 : // div: 0.75
                (op == 2'b10) ? 12'b000000000000 : // exp: 0
                (op == 2'b11) ? 12'b000011000000 : {`MAC_BW{1'b0}} ; // log: 0.75

assign var    = (op == 2'b01) ? point_sub_x : // div: POINT - x_norm
                (op == 2'b10) ? {4{x[11]}, x_frac[7:0]}  : // exp: X_frac
                (op == 2'b11) ? point_sub_x : {`MAC_BW{1'b0}} ; // log: POINT - x_norm  

mac U_mac(.clk(clk),
          .rst_n(rst_n),
          .iA(macX), // macN, and coeff for first cycle
          .iB(macY), // var, and scale for last cycle
          .iC(macZ), // coeff, and offset for last cycle
          .oC(macO)
          );

assign macX = (op == 0) ? X : (fisrt_cycle ? coeff : macO);
assign macY = (op == 0) ? Y : (fisrt_cycle ? var : scale);
assign macZ = (op == 0) ? (acc_en ? macO : Z) : (last_cycle ? offset : coeff);

endmodule



