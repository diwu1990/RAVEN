`include "param_def.sv"

module mux (
    input logic [1:0] mode,
    input logic [`MAC_BW-1 : 0] oC_mac [63 : 0],
    input logic [`MAC_BW-1 : 0] oC_div [63 : 0],
    input logic [`MAC_BW-1 : 0] oC_exp [63 : 0],
    input logic [`MAC_BW-1 : 0] oC_log [63 : 0],
    output logic [`MAC_BW-1 : 0] oC [63 : 0]
);
    
    genvar i;
    generate
        for (i = 0; i < 64; i++) begin
            always_comb begin : proc_oC
                case (mode)
                    2'd0: oC[i] = oC_mac[i];
                    2'd1: oC[i] = oC_div[i];
                    2'd2: oC[i] = oC_exp[i];
                    2'd3: oC[i] = oC_log[i];
                endcase
            end
        end
    endgenerate

endmodule