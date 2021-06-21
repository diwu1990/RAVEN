# %%
import torch
from RAVEN.pe.appr_utils import *
import matplotlib.pyplot as plt

# %%
"""
# Test RoundingNoGrad
"""

# %%
input = torch.tensor([0.51, 0.49, -0.51, -0.49])
print("ceil :", RoundingNoGrad.apply(input, "ceil"))
print("floor:", RoundingNoGrad.apply(input, "floor"))
print("round:", RoundingNoGrad.apply(input, "round"))
print("round:", RoundingNoGrad.apply(input))

# %%
"""
# Test Trunc
"""

# %%
input = torch.tensor([0.751, 0.749, -0.751, -0.749, 0.759, 0.741, -0.759, -0.741])
print(Trunc(input, intwidth=1, fracwidth=2, rounding="ceil"))
print(Trunc(input, intwidth=1, fracwidth=2, rounding="floor"))
print(Trunc(input, intwidth=1, fracwidth=2, rounding="round"))

# %%
"""
# Test Trunc_val
"""

# %%
input = [0.751, 0.749, -0.751, -0.749, 0.759, 0.741, -0.759, -0.741]
print([Trunc_val(i, intwidth=1, fracwidth=2, rounding="ceil") for i in input])
print([Trunc_val(i, intwidth=1, fracwidth=2, rounding="floor") for i in input])
print([Trunc_val(i, intwidth=1, fracwidth=2, rounding="round") for i in input])
