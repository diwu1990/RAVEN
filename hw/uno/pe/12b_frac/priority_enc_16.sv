module priority_enc_16 (
    input logic [15 : 0] in,
    output logic [3 : 0] out
);
    
    assign out = (in[15] == 1'b1) ? 4'd16 :
                 (in[14] == 1'b1) ? 4'd15 :
                 (in[13] == 1'b1) ? 4'd14 :
                 (in[12] == 1'b1) ? 4'd13 :
                 (in[11] == 1'b1) ? 4'd12 :
                 (in[10] == 1'b1) ? 4'd11 :
                 (in[9] == 1'b1)  ? 4'd10 :
                 (in[8] == 1'b1)  ? 4'd09 :
                 (in[7] == 1'b1)  ? 4'd08 :
                 (in[6] == 1'b1)  ? 4'd07 :
                 (in[5] == 1'b1)  ? 4'd06 :
                 (in[4] == 1'b1)  ? 4'd05 :
                 (in[3] == 1'b1)  ? 4'd04 :
                 (in[2] == 1'b1)  ? 4'd03 :
                 (in[1] == 1'b1)  ? 4'd02 :
                 (in[0] == 1'b1)  ? 4'd01 : 4'd00;
                 
endmodule