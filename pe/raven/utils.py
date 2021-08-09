import torch
import math
from RAVEN.pe.appr_utils import Trunc

def poly(coeff, 
            intwidth, 
            fracwidth, 
            x, 
            rounding="round"):
    """
    Calculate the polynomial: output = sum(coeff * x^power)
    "coeff", "intwidth" and "fracwidth" is a 1-D tensor, while "x" is an arbitrary tensor.
    "coeff", "intwidth" and "fracwidth" order go from low to high.
    "coeff" has size N+1 for a degree-N polynomial, e.g., having index of [0, N].
    "intwidth" and "fracwidth" have size N for a degree-N polynomial, e.g., having index of [0, N-1].
    The data bitwidth can be expressed as (1 + "intwidth" + "fracwidth") with 1 for the sign bit.
    "rounding" indicates the rounding mode.
    """
    
    # 1) For multiplication,
    # each input has the format of (1, intwidth, fracwidth),
    # the output has the format of (1, 2 * intwidth + 1, 2 * fracwidth)
    # 2) For accumulation,
    # the input has the format of (1, 2 * intwidth + 1, 2 * fracwidth)
    # the output has the format of (1, 2 * intwidth + 2, 2 * fracwidth)

    poly_degree = len(coeff) - 1

    # prepare the data for the highest order
    mac = coeff[poly_degree]
    var = x
    
    for idx in range(poly_degree):
        curr_idx_minus_1 = poly_degree - idx - 1
        mac = Trunc(mac, intwidth = intwidth[curr_idx_minus_1], fracwidth = fracwidth[curr_idx_minus_1], rounding = rounding)
        var = Trunc(var, intwidth = intwidth[curr_idx_minus_1], fracwidth = fracwidth[curr_idx_minus_1], rounding = rounding)
        prod = mac * var

        c = Trunc(coeff[curr_idx_minus_1],   intwidth = 2 * intwidth[curr_idx_minus_1] + 1, fracwidth = 2 * fracwidth[curr_idx_minus_1], rounding = rounding)
        mac = c + prod
    
    return mac
    
