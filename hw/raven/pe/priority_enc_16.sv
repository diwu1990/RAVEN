module priority_enc_16 (
    input clk,
    input rst_n,
    input logic [15 : 0] in,
    output logic [4 : 0] out
);
    
    assign out = (in[15] == 1'b1) ? 5'd16 :
                 (in[14] == 1'b1) ? 5'd15 :
                 (in[13] == 1'b1) ? 5'd14 :
                 (in[12] == 1'b1) ? 5'd13 :
                 (in[11] == 1'b1) ? 5'd12 :
                 (in[10] == 1'b1) ? 5'd11 :
                 (in[09] == 1'b1) ? 5'd10 :
                 (in[08] == 1'b1) ? 5'd09 :
                 (in[07] == 1'b1) ? 5'd08 :
                 (in[06] == 1'b1) ? 5'd07 :
                 (in[05] == 1'b1) ? 5'd06 :
                 (in[04] == 1'b1) ? 5'd05 :
                 (in[03] == 1'b1) ? 5'd04 :
                 (in[02] == 1'b1) ? 5'd03 :
                 (in[01] == 1'b1) ? 5'd02 :
                 (in[00] == 1'b1) ? 5'd01 : 5'd00;
                 
endmodule