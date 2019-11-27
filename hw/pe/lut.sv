module lut(a,out);
	parameter BW = 8;
    input [BW-1:0] a;
    output reg [BW-1:0] out;

always @(a[BW-1:BW-4]) begin
case(a[BW-1:BW-4]) //random numbers
    4'b0000 : out = 2;
    4'b0001 : out = 3;
    4'b0010 : out = 4;
    4'b0011 : out = 7; 
    4'b0100 : out = 2;
    4'b0101 : out = 3;
    4'b0110 : out = 4;
    4'b0111 : out = 7;
	default: out = 1;
endcase
end

endmodule
