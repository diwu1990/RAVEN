import torch

class RoundingNoGrad(torch.autograd.Function):
    """
    RoundingNoGrad is a rounding operation which bypasses the input gradient to output directly.
    Original round()/floor()/ceil() opertions have a gradient of 0 everywhere, which is not useful when doing approximate computing.
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
            raise ValueError("Rounding mode is not supported.")
    
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
    max_val = (2**(intwidth + fracwidth) - 1) / scale
    min_val = 0 - (2**(intwidth + fracwidth)) / scale
    return RoundingNoGrad.apply(input.mul(scale), rounding).div(scale).clamp(min_val, max_val)
    
    
def Appr_Taylor(scale, const, var, coeff, power, sign, fxp=True, intwidth=7, fracwidth=8, rounding="floor", keepwidth=True):
    """
    Calculate the result of approximate Taylor series.
    The results is calculated as:
    output = scale * (const + sum(coeff * var^power * sign))
    
    The approximate Taylor series require following rules for hardware efficiency.
    1) Tensor scale is calculated by shifting input or using very small LUT.
    2) Tensor const is calculated using LUT
    3) Tensor var is calulated as (input - offset)
    4) Power[i] == 1, when i == 0
    5) Power[i] == either power[i-1] or power[i-1]+1, when i > 0
    6) Coeff[i] == 2^k, where k is a positive/negtive integer.
    
    "fxp" means whether to performance fixed point calculation, in which the data bitwidth can be expressed as
    (1 + "intwidth" + "fracwidth"), and "rounding" indicates the rounding mode.
    """
    
    def flp_poly(scale, const, var, coeff, power, sign):
        # initialization
        acc = torch.zeros_like(var).add(const)
        
        # cumulate the terms
        for idx in range(0, len(coeff)):
            acc.add_(coeff[idx] * var.pow(power[idx]) * sign[idx])
            
        return acc.mul(scale)
    
    def fxp_poly(scale, const, var, coeff, power, sign, intwidth=7, fracwidth=8, rounding="floor", keepwidth=True):
        # 1) For multiplication,
        # each input has the format of (1, intwidth, fracwidth),
        # the output has the format of (1, 2 * intwidth + 1, 2 * fracwidth)
        # 2) For accumulation,
        # the input has the format of (1, 2 * intwidth + 1, 2 * fracwidth)
        # the output has the format of (1, 2 * intwidth + 2, 2 * fracwidth)

        acc = torch.zeros_like(var)
        prod = torch.ones_like(var)
        shift = torch.zeros_like(var)
        output = torch.zeros_like(var)
        
        # both scale and var are of format (1, intwidth, fracwidth), as they participate in the multiplication
        scale = Trunc(scale, 
                      intwidth = intwidth, 
                      fracwidth = fracwidth, 
                      rounding = rounding)
        
        var = Trunc(var, 
                    intwidth = intwidth, 
                    fracwidth = fracwidth, 
                    rounding = rounding)
        
        # const is of format (1, 2 * intwidth + 1, 2 * fracwidth), as it participates in the accumulation
        const = Trunc(const, 
                      intwidth = 2 * intwidth + 2, 
                      fracwidth = 2 * fracwidth, 
                      rounding = rounding)
        
        # initialization
        acc = acc.add(const)
        
        # calculate the first term with power value of 1
        shift = Trunc(prod.mul(coeff[0]), 
                      intwidth = intwidth, 
                      fracwidth = fracwidth, 
                      rounding = rounding)
        
        prod = shift.mul(var.pow(power[0]))
        
        acc = acc.add(prod.mul(sign[0]))
        
        # cumulate the rest terms
        for idx in range(1, len(coeff)):
            shift = Trunc(prod.mul(coeff[idx]/coeff[idx-1]), 
                          intwidth = intwidth, 
                          fracwidth = fracwidth, 
                          rounding = rounding)
            
            prod = shift.mul(var.pow(power[idx]-power[idx-1]))
            
            acc = Trunc(acc.add(prod.mul(sign[idx])), 
                        intwidth = 2 * intwidth + 2, 
                        fracwidth = 2 * fracwidth, 
                        rounding = rounding)
            
        output = Trunc(acc, 
                       intwidth = intwidth, 
                       fracwidth = fracwidth, 
                       rounding = rounding).mul(scale)
        
        if keepwidth is True:
            # output has the same bitwidth as input var
            return output
        elif keepwidth is False:
            # maintain the bitwidth of multiplier
            # bitwidth of output is twice that of input var
            return Trunc(output, 
                         intwidth = 2 * intwidth + 1, 
                         fracwidth = 2 * fracwidth, 
                         rounding = rounding)
        else:
            raise ValueError("Input keepwidth need to be of bool type.")
    
    if fxp is True:
        return fxp_poly(scale, const, var, coeff, power, sign, intwidth, fracwidth, rounding)
    elif fxp is False:
        return flp_poly(scale, const, var, coeff, power, sign)
    else:
        raise ValueError("fxp mode have to be of bool type.")
        