`include "param_def.sv"
`include "array.sv"
`include "nonlinear.sv"
`include "adder_tree.sv"
`include "mux.sv"
`include "demux.sv"

module simd_array_mac (
    input logic clk,    // Clock
    input logic rst_n,  // Asynchronous reset active low
    input logic [1:0] mode,
    input logic [`MAC_BW-1 : 0] iA [63 : 0],
    input logic [`MAC_BW-1 : 0] iB [63 : 0],
    output logic [2*`MAC_BW-1 : 0] oL1 [15 : 0],
    output logic [2*`MAC_BW-1 : 0] oL2 [ 3 : 0],
    output logic [2*`MAC_BW-1 : 0] oL3
);
    logic [`MAC_BW-1 : 0] oC [63 : 0];
    
    logic [`MAC_BW-1 : 0] iA_mac [63 : 0];
    logic [`MAC_BW-1 : 0] iB_mac [63 : 0];
    logic [`MAC_BW-1 : 0] oC_mac [63 : 0];

    logic [`MAC_BW-1 : 0] iA_nl [63 : 0];
    logic [`MAC_BW-1 : 0] iB_nl [63 : 0];
    logic [`MAC_BW-1 : 0] oC_nl [63 : 0];


    demux U_demux(
        .mode(mode),
        .iA(iA),
        .iB(iB),
        .iA_mac(iA_mac),
        .iB_mac(iB_mac),
        .iA_nl(iA_nl),
        .iB_nl(iB_nl),
        );

    mux U_mux(
        .mode(mode),
        .iC_mac(iC_mac),
        .iC_nl(iC_nl),
        .iC(iC),
        );

    array U_array(
        .clk(clk),
        .rst_n(rst_n),
        .iA(iA_mac),
        .iB(iB_mac),
        .oC(oC_mac)
        );

    nonlinear U_nonlinear(
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .iA(iA_nl),
        .iB(iB_nl),
        .oC(oC_nl)
        );

    adder_tree U_adder_tree(
        .clk(clk),
        .rst_n(rst_n),
        .in(oC),
        .out_l1(oL1),
        .out_l2(oL2),
        .out_l3(oL3)
        );

endmodule