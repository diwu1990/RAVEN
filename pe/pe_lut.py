import torch

class PE_LUT(torch.nn.Module):
    def __init__(self, distribution=None):
        super(PE_LUT, self).__init__()
        self.div_lut = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.div_max = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)

        self.exp_lut = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.exp_max = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)

        self.log_lut = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.log_max = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        
        if distribution is "left":
            pass
        elif distribution is "middle":
            pass
        elif distribution is "right":
            pass
        elif distribution is "sides":
            pass
        elif distribution is "uniform":
            pass
        else:
            raise ValueError("PE_LUT distribution is not implemented.")

    def forward(self):
        return self.div_lut, self.div_max, self.exp_lut, self.exp_max, self.log_lut, self.log_max
