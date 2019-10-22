import torch
from RAVEN.pe.pe_lut import PE_LUT

class RAVEN_PE(torch.nn.Module):
    """
    reconfigurable processing element for RAVEN.
    
    supported function:
    1. add
    2. mul
    3. div
    4. exp
    5. log
    
    supported approximation algorithms:
    1. taylor series based
    2. log2 based
    """
    def __init__(self, bitwidth=8, intwidth=4, distribution="middle"):
        super(RAVEN_PE, self).__init__()
        self.bitwidth = torch.nn.Parameter(torch.tensor([bitwidth]), requires_grad=False)
        self.intwidth = torch.nn.Parameter(torch.tensor([intwidth]), requires_grad=False)
        self.fracwidth = self.bitwidth - self.intwidth
        pe_lut = PE_LUT(distribution)
        self.div_lut, self.exp_lut, self.log_lut = pe_lut()
    
    # taylor series based implementation
    def taylor_add(self, in_1, in_2, cycle):
        return in_1.add(in2)
    
    def taylor_mul(self, in_1, in_2, cycle):
        return in_1.mul(in2)
    
    def taylor_div(self, in_1, in_2, cycle):
        self.div_lut
        pass
    
    def taylor_exp(self, in_1, cycle):
        self.exp_lut
        pass
    
    def taylor_log(self, in_1, cycle):
        self.log_lut
        pass

    # log2 based implementation
    def log2_add(self, in_1, in_2):
        pass
    
    def log2_mul(self, in_1, in_2):
        pass
    
    def log2_div(self, in_1, in_2):
        self.div_lut
        pass
    
    def log2_exp(self, in_1):
        self.exp_lut
        pass
    
    def log2_log(self, in_1):
        self.log_lut
        pass
    
    def forward(self, in_1, in_2, function="add", mode="taylor", cycle=1):
        if function == "add":
            return self.add(in_1, in_2)
        elif function == "mul":
            return self.mul(in_1, in_2)
        elif function == "div":
            return self.div(in_1, in_2, cycle)
        elif function == "exp":
            return self.exp(in_1, cycle)
        elif function == "log":
            return self.log(in_1, cycle)
        else:
            raise ValueError("Input functional mode is not implemented.")
            