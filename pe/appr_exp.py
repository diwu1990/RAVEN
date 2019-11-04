import torch
from RAVEN.pe.appr_poly import Poly_Appr_Taylor
    
class APPR_EXP(torch.nn.Module):
    """
    Approximate exponentiation
    """
    def __init__(self, bitwidth=8, mode="taylor", distribution="uniform"):
        super(APPR_EXP, self).__init__()
        self.bitwidth = bitwidth
        self.mode = mode
        self.distribution = distribution
        self.coeff, self.power, self.sign = self.exp_taylor_lut(self.bitwidth, self.distribution)
        # check the size of coeff, power and sign are the same.
        assert (self.coeff.size() == self.power.size() and self.power.size() == self.sign.size()), "Non-indentical size for coeff, power and sign."
    
    def exp_taylor_lut(self, bitwidth=8, distribution="uniform"):
        if self.distribution == "uniform"
            if self.bitwidth == 8:
                coeff = None
                power = None
                sign = None
                return coeff, power, sign
            
            if self.bitwidth == 12:
                coeff = None
                power = None
                sign = None
                return coeff, power, sign
            
            if self.bitwidth == 16:
                coeff = None
                power = None
                sign = None
                return coeff, power, sign
        
    def forward(self, input)
        return Poly_Appr_Taylor(input, self.coeff, self.power, self.sign)
    
    