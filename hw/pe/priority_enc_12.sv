module priority_enc_12 (
    input logic [11 : 0] in,
    output logic [3 : 0] out
);
    
    assign out = (in[11] == 1'b1) ? 4'd12 :
                 (in[10] == 1'b1) ? 4'd10 :
                 (in[9] == 1'b1) ? 4'd09 :
                 (in[8] == 1'b1) ? 4'd08 :
                 (in[7] == 1'b1) ? 4'd07 :
                 (in[6] == 1'b1) ? 4'd06 :
                 (in[5] == 1'b1) ? 4'd05 :
                 (in[4] == 1'b1) ? 4'd04 :
                 (in[3] == 1'b1) ? 4'd03 :
                 (in[2] == 1'b1) ? 4'd02 :
                 (in[1] == 1'b1) ? 4'd01 :
                 (in[0] == 1'b1) ? 4'd00 : 4'd00;
                 
endmodule