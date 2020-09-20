`include "param_def.sv"

module mux (
    input logic [1:0] mode,
    input logic [`MAC_BW-1 : 0] oC_mac [63 : 0],
    input logic [`MAC_BW-1 : 0] oC_nl [63 : 0],
    output logic [`MAC_BW-1 : 0] oC [63 : 0]
);
    
    genvar i;
    generate
        for (i = 0; i < 64; i++) begin
            assign oC[i] = (mode == 0) ? oC_mac[i] : oC_nl[i];
        end
    endgenerate

endmodule