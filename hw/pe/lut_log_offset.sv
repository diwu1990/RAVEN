module lut_log_offset (
    input logic [3:0] shift_offset,
    output logic [23:0] log_offset
);

    always_comb begin : proc_log_offset
        case (shift_offset)
            4'b0000 : log_offset <= 12'b0;
            4'b0001 : log_offset <= 12'b0;
            4'b0010 : log_offset <= 12'b101010011000101010011000;
            4'b0011 : log_offset <= 12'b111101000100111101000100;
            4'b0100 : log_offset <= 12'b100110100010100110100010;
            4'b0101 : log_offset <= 12'b110010101010100100000110;
            4'b0110 : log_offset <= 12'b110010101010100100000110;
            4'b0111 : log_offset <= 24'b110010101010100100000110;
            4'b1000 : log_offset <= 12'b000000010110000000010110;
            4'b1001 : log_offset <= 12'b000000111110000000111110;
            4'b1010 : log_offset <= 12'b000010100010000010100010;
            4'b1011 : log_offset <= 12'b000110111010000110111010;
            4'b1100 : log_offset <= 12'b010010110000010010110000;
            4'b1101 : log_offset <= 12'b110010111111110010111111;
            4'b1110 : log_offset <= 12'b111111111111111111111111;
            4'b1111 : log_offset <= 12'b111111111111111111111111;
            default : log_offset <= 12'b000100000000000100000000;
        endcase
    end

endmodule