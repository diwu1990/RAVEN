module lut_log_offset (
    input logic clk,
    input logic rst_n,
    input logic [4:0] shift_offset,
    output logic [39:0] log_offset
);
    
    always_ff @(posedge clk or negedge rst_n) begin : proc_log_offset
        if(~rst_n) begin
            log_offset <= 0;
        end else begin
            case (shift_offset)
                4'b0000 : log_offset <= 40'b0;
                4'b0001 : log_offset <= 40'b0;
                4'b0010 : log_offset <= 40'b110101001100010101001100000000000;
                4'b0011 : log_offset <= 40'b1111110100010011110100010000000000;
                4'b0100 : log_offset <= 40'b11110011010001010011010001000000000;
                4'b0101 : log_offset <= 40'b1111011001010101010010000011000000000;
                4'b0110 : log_offset <= 40'b10111001010101010010000011000000000;
                4'b0111 : log_offset <= 40'b10011001010101010010000011000000000;
                4'b1000 : log_offset <= 40'b11010100000001011000000001011000000000;
                4'b1001 : log_offset <= 40'b00000011111000000011111000000000;
                4'b1010 : log_offset <= 40'b00100001010001000001010001000000000;
                4'b1011 : log_offset <= 40'b00011011101000011011101000000000;
                4'b1100 : log_offset <= 40'b01001011000001001011000000000000;
                4'b1101 : log_offset <= 40'b11001011111111001011111100000000;
                4'b1110 : log_offset <= 40'b101011111111111111111111111100000000;
                4'b1111 : log_offset <= 40'b11111111111111111111111100000000;
                default : log_offset <= 40'b1111110000010000000000010000000000000000;
            endcase
        end
    end
    // always_comb begin : proc_log_offset
    //     case (shift_offset)
    //         4'b0000 : log_offset <= 24'b0;
    //         4'b0001 : log_offset <= 24'b0;
    //         4'b0010 : log_offset <= 24'b101010011000101010011000;
    //         4'b0011 : log_offset <= 24'b111101000100111101000100;
    //         4'b0100 : log_offset <= 24'b100110100010100110100010;
    //         4'b0101 : log_offset <= 24'b110010101010100100000110;
    //         4'b0110 : log_offset <= 24'b110010101010100100000110;
    //         4'b0111 : log_offset <= 24'b110010101010100100000110;
    //         4'b1000 : log_offset <= 24'b000000010110000000010110;
    //         4'b1001 : log_offset <= 24'b000000111110000000111110;
    //         4'b1010 : log_offset <= 24'b000010100010000010100010;
    //         4'b1011 : log_offset <= 24'b000110111010000110111010;
    //         4'b1100 : log_offset <= 24'b010010110000010010110000;
    //         4'b1101 : log_offset <= 24'b110010111111110010111111;
    //         4'b1110 : log_offset <= 24'b111111111111111111111111;
    //         4'b1111 : log_offset <= 24'b111111111111111111111111;
    //         default : log_offset <= 24'b000100000000000100000000;
    //     endcase
    // end

endmodule