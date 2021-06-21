import torch
import math
import matplotlib.pyplot as plt

# This file is for UNO PE.

class RoundingNoGrad(torch.autograd.Function):
    """
    RoundingNoGrad is a rounding operation which bypasses the input gradient to output directly.
    Original round()/floor()/ceil() opertions have a gradient of 0 everywhere, which is not useful 
    when doing approximate computing.
    This is something like the straight-through estimator (STE) for quantization-aware training.
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, mode="round"):
        if mode == "round":
            return input.round()
        elif mode == "floor":
            return input.floor()
        elif mode == "ceil":
            return input.ceil()
        else:
            raise ValueError("Input rounding is not supported.")
    
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None
    
    
def Trunc(input, intwidth=7, fracwidth=8, rounding="floor"):
    """
    Trunc is an operation to convert data to format (1, intwidth, fracwidth).
    """
    scale = 2**fracwidth
    max_val = (2**(intwidth + fracwidth) - 1)
    min_val = 0 - (2**(intwidth + fracwidth))
    return RoundingNoGrad.apply(input.mul(scale), rounding).clamp(min_val, max_val).div(scale)
    
    
def Trunc_val(input, intwidth=7, fracwidth=8, rounding="round"):
    """
    Trunc_val is an operation to convert one single value to format (1, intwidth, fracwidth).
    """
    scale = 2**fracwidth
    max_val = (2**(intwidth + fracwidth) - 1)
    min_val = 0 - (2**(intwidth + fracwidth))
    if rounding == "round":
        return max(min(round(input*scale), max_val), min_val)/scale
    elif rounding == "floor":
        return max(min(math.floor(input*scale), max_val), min_val)/scale
    elif rounding == "ceil":
        return max(min(math.ceil(input*scale), max_val), min_val)/scale
    else:
        raise ValueError("Input rounding is not supported.")

