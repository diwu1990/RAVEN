`include "param_def.sv"
`include "mac.sv"

module uno (
    input clk,    // Clock
    input rst_n,  // Asynchronous reset active low
    input XSEL,
    input [`MAC_BW-1 : 0] ifm,
    input [`MAC_BW-1 : 0] ifm_var,
    input [`MAC_BW-1 : 0] ifm_scale,
    input [2*`MAC_BW-1 : 0] i_sum,
    input YSEL,
    input [`MAC_BW-1 : 0] weight,
    input [`MAC_BW-1 : 0] sum_y,
    input [`MAC_BW-1 : 0] ifm_coeff_0,
    input ZSEL,
    input [2*`MAC_BW-1 : 0] ifm_coeff_1,
    input [2*`MAC_BW-1 : 0] ifm_offset,
    output logic [`MAC_BW-1 : 0] o_X,
    output logic [`MAC_BW-1 : 0] o_Y,
    output logic [2*`MAC_BW-1 : 0] o_sum
);

    logic [`MAC_BW-1 : 0] X,
    logic [`MAC_BW-1 : 0] Y,
    logic [`MAC_BW-1 : 0] Z,

    always_comb begin : proc_X
        case (XSEL):
            2'b00 : X <= ifm;
            2'b01 : X <= ifm_var;
            2'b10 : X <= ifm_scale;
            2'b11 : X <= sum_x;
            default : X <= ifm;
        endcase
    end

    always_comb begin : proc_Y
        case (YSEL):
            2'b00 : Y <= weight;
            2'b01 : Y <= sum_y;
            2'b10 : Y <= ifm_coeff_0;
            2'b11 : Y <= 1;
            default : Y <= weight;
        endcase
    end

    always_comb begin : proc_Z
        case (YSEL):
            2'b00 : Z <= ifm_coeff_1;
            2'b01 : Z <= ifm_offset;
            2'b10 : Z <= sum_z;
            2'b11 : Z <= sum;
            default : Z <= sum_o;
        endcase
    end

    mac U_mac(.clk(clk),
              .rst_n(rst_n),
              .iA(X), // macN, and coeff for first cycle
              .iB(Y), // var_x, and scale for last cycle
              .iC(Z), // coeff, and offset for last cycle
              .oC(o_sum)
              );

    assign o_O = macO;

endmodule



