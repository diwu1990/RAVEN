`include "param_def.sv"

module demux (
    input logic [1:0] mode,
    input logic [`MAC_BW-1 : 0] iA [63 : 0],
    input logic [`MAC_BW-1 : 0] iB [63 : 0],
    output logic [`MAC_BW-1 : 0] iA_mac [63 : 0],
    output logic [`MAC_BW-1 : 0] iB_mac [63 : 0],
    output logic [`MAC_BW-1 : 0] iA_nl [63 : 0],
    output logic [`MAC_BW-1 : 0] iB_nl [63 : 0]
);
    
    genvar i;
    generate
        for (i = 0; i < 64; i++) begin
            assign iA_mac[i] = (mode == 0) ? iA[i] : 0;
            assign iB_mac[i] = (mode == 0) ? iB[i] : 0;

            assign iA_nl[i] = (mode != 0) ? iA[i] : 0;
            assign iB_nl[i] = (mode != 0) ? iB[i] : 0;
        end
    endgenerate

endmodule