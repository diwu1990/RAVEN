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
        self.bitwidth = bitwidth
        self.mode = mode
        self.div_lut = div_lut
        self.exp_lut = exp_lut
        self.log_lut = log_lut

    def forward():