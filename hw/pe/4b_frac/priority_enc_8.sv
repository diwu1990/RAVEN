module priority_enc_8 (
    input logic [7 : 0] in,
    output logic [3 : 0] out
);
    
    assign out = (in[7] == 1'b1) ? 4'd08 :
                 (in[6] == 1'b1) ? 4'd07 :
                 (in[5] == 1'b1) ? 4'd06 :
                 (in[4] == 1'b1) ? 4'd05 :
                 (in[3] == 1'b1) ? 4'd04 :
                 (in[2] == 1'b1) ? 4'd03 :
                 (in[1] == 1'b1) ? 4'd02 :
                 (in[0] == 1'b1) ? 4'd01 : 4'd00;
                 
endmodule