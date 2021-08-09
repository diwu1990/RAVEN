# %%
import torch
from RAVEN.pe.raven.utils import poly
import matplotlib.pyplot as plt

# %%
"""
# test poly
"""

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a = torch.arange(0, 1, 0.001).to(device)

precise = torch.exp(a).to(device)

var = a

coeff = torch.Tensor([1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]).to(device)
intwidth  = torch.ones(8).to(device) * 3
fracwidth = torch.ones(8).to(device) * 5
coeff = coeff[0:8]

approximate = poly(coeff, 
                    intwidth, 
                    fracwidth, 
                    var, 
                    rounding="round")

error = (approximate - precise) / precise
print("relative error:")
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())

error = (approximate - precise)
print("absolute error:")
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())
