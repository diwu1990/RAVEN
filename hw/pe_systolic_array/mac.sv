`include "param_def.sv"

module mac (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [`MAC_BW-1 : 0] iA,
    input logic [`MAC_BW-1 : 0] iB,
    input logic [2*`MAC_BW-1 : 0] iC,
    output logic [2*`MAC_BW-1 : 0] oC
);
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_output
        if(~rst_n) begin
            oC <= 0;
        end else begin
            oC <= iA * iB + iC;
        end
    end

endmodule