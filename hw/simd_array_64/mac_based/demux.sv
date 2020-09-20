`include "param_def.sv"

module demux (
    input logic [1:0] mode,
    input logic [`MAC_BW-1 : 0] iA [63 : 0],
    input logic [`MAC_BW-1 : 0] iB [63 : 0],
    output logic [`MAC_BW-1 : 0] iA_mac [63 : 0],
    output logic [`MAC_BW-1 : 0] iB_mac [63 : 0],
    output logic [`MAC_BW-1 : 0] iA_div [63 : 0],
    output logic [`MAC_BW-1 : 0] iB_div [63 : 0],
    output logic [`MAC_BW-1 : 0] iA_exp [63 : 0],
    output logic [`MAC_BW-1 : 0] iB_exp [63 : 0],
    output logic [`MAC_BW-1 : 0] iA_log [63 : 0],
    output logic [`MAC_BW-1 : 0] iB_log [63 : 0],
);
    
    genvar i;
    generate
        for (i = 0; i < 64; i++) begin
            assign iA_mac[i] = (mode == 0) ? iA[i] : 0;
            assign iB_mac[i] = (mode == 0) ? iB[i] : 0;

            assign iA_div[i] = (mode == 1) ? iA[i] : 0;
            assign iB_div[i] = (mode == 1) ? iB[i] : 0;

            assign iA_exp[i] = (mode == 2) ? iA[i] : 0;
            assign iB_exp[i] = (mode == 2) ? iB[i] : 0;

            assign iA_log[i] = (mode == 3) ? iA[i] : 0;
            assign iB_log[i] = (mode == 3) ? iB[i] : 0;
        end
    endgenerate

endmodule