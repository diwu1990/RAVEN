# %%
import torch
from RAVEN.pe.uno.utils import *
import matplotlib.pyplot as plt

# %%
"""
# test UNOtaylor
"""

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a = torch.arange(0, 1, 0.001).to(device)

precise = torch.exp(a).to(device)

point = 0.5
scale = torch.exp(torch.tensor([point])).to(device)
var = a - point

coeff = [1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]

coeff = coeff[0:8]
print(coeff)

approximate = UNOtaylor(scale, 
                         coeff, 
                         var, 
                         fxp=True, 
                         intwidth=7, 
                         fracwidth=8, 
                         rounding_coeff="round", 
                         rounding_var="floor", 
                         keepwidth=True)

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())

# %%
"""
# test data_gen
"""

# %%
data = data_gen(data_range="0.0_1.0", mu=0.7, sigma = 1)

# %%

# %%
"""
# point search test
"""

# %%
"""
## flp
"""

# %%
intwidth = 7
fracwidth = 8
point_search(func="div", fxp=False, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="exp", fxp=False, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="log", fxp=False, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)

# %%
"""
## fxp
"""

# %%
intwidth = 7
fracwidth = 4
point_search(func="div", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="exp", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="log", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)

# %%
intwidth = 7
fracwidth = 8
point_search(func="div", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="exp", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="log", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)

# %%
intwidth = 7
fracwidth = 12
point_search(func="div", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="exp", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="log", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)

# %%
intwidth = 7
fracwidth = 16
point_search(func="div", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="exp", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)
point_search(func="log", fxp=True, intwidth=intwidth, fracwidth=fracwidth, 
             valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True)

# %%
