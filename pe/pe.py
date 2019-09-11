import torch

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
    def __init__(self, bitwidth=8, mode="taylor", div_lut=None, exp_lut=None, log_lut=None):
        super(RAVEN_PE, self).__init__()
        self.bitwidth = torch.nn.Parameter(torch.tensor([bitwidth]), requires_grad=False)
        self.mode = mode
        self.div_lut = div_lut
        self.exp_lut = exp_lut
        self.log_lut = log_lut
    
    def mul(self, in_1, in_2, cycle):
        pass
    
    def add(self, in_1, in_2, cycle):
        pass
    
    def div(self, in_1, in_2, cycle):
        self.div_lut
        pass
    
    def exp(self, in_1, cycle):
        self.exp_lut
        pass
    
    def log(self, in_1, cycle):
        self.log_lut
        pass
    
    def forward(self, in_1, in_2, function="add", cycle=1):
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
            