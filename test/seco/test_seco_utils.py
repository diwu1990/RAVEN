# %%
import torch
from RAVEN.pe.seco.utils import *
import matplotlib.pyplot as plt

# %%
"""
# Test for SECO_Taylor
"""

# %%
a = torch.arange(0, 1, 0.001).cuda()
a_int = a.floor()

precise = torch.exp(a)

point = torch.tensor(0.)
scale = torch.exp(a_int + point)
const = torch.tensor(1.0)
input = a - point

coeff = [1/1, 1/2, 1/4, 1/8, 1/16, 1/32]
power = [1  ,   2,   3,   4,    5,    5]
sign  = [1  ,   1,   1,  -1,    1,    1]

approximate = SECO_Taylor(scale, 
                          const, 
                          input, 
                          coeff, 
                          power, 
                          sign, 
                          fxp=True, 
                          intwidth=7, 
                          fracwidth=8, 
                          rounding="round", 
                          keepwidth=True)

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())
print()

# %%
"""
# exp parameter generation
"""

# %%
"""
1. exp_data_gen test
"""

# %%
data = exp_data_gen("left")
fig = plt.hist(data.cpu().numpy(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for data")
plt.show()

# %%
"""
2. exp_param_gen test
"""

# %%
exp_param_gen(distribution="uniform", intwidth=7, fracwidth=8, rounding="floor", keepwidth=True, valid=True)

