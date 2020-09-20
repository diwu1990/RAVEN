`include "param_def.sv"
`include "mac.sv"

module array (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [1:0] mode,
    input logic [`MAC_BW-1 : 0] iA [63 : 0],
    input logic [`MAC_BW-1 : 0] iB [63 : 0],
    output logic [2*`MAC_BW-1 : 0] oC [63 : 0]
);
    
    genvar i;
    generate
        for (i = 0; i < 64; i++) begin
            mac U_mac(
                .clk(clk),
                .rst_n(rst_n),
                .iA(iA[i]),
                .iB(iB[i]),
                .iC(oC[i]),
                .oC(oC[i])
                );
        end
    endgenerate

endmodule