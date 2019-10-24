import torch

class RAVEN_EXP(torch.nn.Module):
    """
    Exponentiation in RAVEN
    
    supported approximation algorithms:
    1. taylor series based
    2. log2 based
    """
    def __init__(self, fxp=True, bitwidth=8, intwidth=4, mode="taylor", distribution="middle"):
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
    
    # log2 based implementation
    def log2_exp(self, in_1):
        self.exp_lut
        pass
    
    def forward(self, input, cycle=1):
        if self.mode == "taylor":
            return self.taylor_exp(input, cycle)
        elif self.mode == "log2":
            return self.log2_log(input, cycle)
        else:
            raise ValueError("Input functional mode is not implemented.")