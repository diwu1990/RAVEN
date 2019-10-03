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
    1. log2 based
    2. taylor series based
    """
    def __init__(self, bitwidth=8, fracwidth=3, distribution=None):
        super(RAVEN_PE, self).__init__()
        self.bitwidth = torch.nn.Parameter(torch.tensor([bitwidth]), requires_grad=False)
        self.fracwidth = torch.nn.Parameter(torch.tensor([fracwidth]), requires_grad=False)
        pe_lut = PE_LUT(distribution)
        self.div_lut, self.exp_lut, self.log_lut = PE_LUT()
    
    # log2 based implementation
    def log2_mul(self, in_1, in_2):
        pass
    
    def log2_add(self, in_1, in_2):
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
    
    # taylor series based implementation
    def taylor_mul(self, in_1, in_2, cycle):
        pass
    
    def taylor_add(self, in_1, in_2, cycle):
        pass
    
    def taylor_div(self, in_1, in_2, cycle):
        self.div_lut
        pass
    
    def taylor_exp(self, in_1, cycle):
        self.exp_lut
        pass
    
    def taylor_log(self, in_1, cycle):
        self.log_lut
        pass
    
    def forward(self, in_1, in_2, function="add", mode="log2", cycle=1):
        if function == "add":
            return self.add(in_1, in_2, cycle)
        elif function == "mul":
            return self.mul(in_1, in_2, cycle)
        elif function == "div":
            return self.div(in_1, in_2, cycle)
        elif function == "exp":
            return self.exp(in_1, cycle)
        elif function == "log":
            return self.log(in_1, cycle)
        else:
            raise ValueError("Input functional mode is not implemented.")
            