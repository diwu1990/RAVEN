module lut_exp_scale (
    input logic clk,
    input logic rst_n,
    input logic [3:0] exp_int,
    output logic [19:0] exp_scale
);
    always_ff @(posedge clk or negedge rst_n) begin : proc_exp_scale
        if(~rst_n) begin
            exp_scale <= 0;
        end else begin
            case (exp_int)
                4'b0000 : exp_scale <= 20'b0001000000000000;
                4'b0001 : exp_scale <= 20'b0010101110000000;
                4'b0010 : exp_scale <= 20'b0111011001000000;
                4'b0011 : exp_scale <= 20'b0111111111110000;
                4'b0100 : exp_scale <= 20'b0111111111110000;
                4'b0101 : exp_scale <= 20'b0111111111110000;
                4'b0110 : exp_scale <= 20'b0111111111110000;
                4'b0111 : exp_scale <= 20'b0111111111110000;
                4'b1000 : exp_scale <= 20'b0000000101100000;
                4'b1001 : exp_scale <= 20'b10100000001111100000;
                4'b1010 : exp_scale <= 20'b0010000101000100000;
                4'b1011 : exp_scale <= 20'b1110001101110100000;
                4'b1100 : exp_scale <= 20'b0000100101100000000;
                4'b1101 : exp_scale <= 20'b101100101111110000;
                4'b1110 : exp_scale <= 20'b011111111111110000;
                4'b1111 : exp_scale <= 20'b11111111111110000;
                default : exp_scale <= 20'b00001000000000000;
            endcase
        end
    end
    // always_comb begin : proc_exp_scale
    //     case (exp_int)
    //         4'b0000 : exp_scale <= 12'b000100000000;
    //         4'b0001 : exp_scale <= 12'b001010111000;
    //         4'b0010 : exp_scale <= 12'b011101100100;
    //         4'b0011 : exp_scale <= 12'b011111111111;
    //         4'b0100 : exp_scale <= 12'b011111111111;
    //         4'b0101 : exp_scale <= 12'b011111111111;
    //         4'b0110 : exp_scale <= 12'b011111111111;
    //         4'b0111 : exp_scale <= 12'b011111111111;
    //         4'b1000 : exp_scale <= 12'b000000010110;
    //         4'b1001 : exp_scale <= 12'b000000111110;
    //         4'b1010 : exp_scale <= 12'b000010100010;
    //         4'b1011 : exp_scale <= 12'b000110111010;
    //         4'b1100 : exp_scale <= 12'b010010110000;
    //         4'b1101 : exp_scale <= 12'b110010111111;
    //         4'b1110 : exp_scale <= 12'b111111111111;
    //         4'b1111 : exp_scale <= 12'b111111111111;
    //         default : exp_scale <= 12'b000100000000;
    //     endcase
    // end
        
endmodule