import torch

class RAVEN_EXP(torch.nn.Module):
    """
    Exponentiation in RAVEN
    
    supported approximation algorithms:
    1. taylor series based
    2. log2 based
    """
    def __init__(self, bitwidth=8, intwidth=4, mode="fxp", distribution="middle"):
        super(RAVEN_PE, self).__init__()
        self.bitwidth = torch.nn.Parameter(torch.tensor([bitwidth]), requires_grad=False)
        self.intwidth = torch.nn.Parameter(torch.tensor([intwidth]), requires_grad=False)
        self.fracwidth = self.bitwidth - self.intwidth
        pe_lut = PE_LUT(distribution)
        self.div_lut, self.exp_lut, self.log_lut = pe_lut()
    
    # taylor series based implementation
    def taylor_exp(self, in_1, cycle):
        self.exp_lut
        pass
    
    def log2_exp(self, in_1):
        self.exp_lut
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