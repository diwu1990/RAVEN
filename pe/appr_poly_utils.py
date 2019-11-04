import torch

def trunc(x, bitwidth, fracwidth):
    x.mul(fracwidth)
    
    
def Poly_Appr_Taylor(x, coeff, power, sign):
    """
    Calculate the result for polynomial.
    """
    acc = torch.zeros_like(x)
    prod = torch.ones_like(x)
    
    for idx in range(len_coeff):
        prod.mul_(x.pow(power[idx]))
        acc.add_(coeff[idx] *  * sign[idx])
        
    return acc
    
class Ref_Func(object):
    """
    This is the reference function to the approximate function
    """
    def __init__(self, func):
        super(Ref_Func, self).__init__()
        self.func = func
    
    def ref_div(self, input0, input1):
        return torch.div(input0, input1)

    def ref_exp(self, input):
        return torch.exp(input)

    def ref_log(self, input):
        return torch.log(input)

    def inst(self):
        if self.func is "div":
            return self.ref_div
        elif self.func is "exp":
            return self.ref_exp
        elif self.func is "log":
            return self.ref_log
        else:
            raise ValueError("Input function \'" + self.func + "\' is not supported in Ref_Func Class.")


class Appr_Func(object):
    """
    This is the approximate function to the refence function.
    It requires two kinds of inputs:
    1) the original inputs
    2) parameter for the approximate polynomials, including coeffient, power and sign of each term.
    For div, exp and log, their approximate Taylor series are different, so we need to specify the functio type explicitly.
    """
    def __init__(self, func):
        super(Appr_Func, self).__init__()
        self.func = func
        
    def appr_div(self, input0, input1, coeff, power, sign):
        return None
    
    def appr_exp(self, input, coeff, power, sign):
        
        return output
    
    def appr_log(self, input, coeff, power, sign):
        return None
    
    def inst(self):
        if self.func is "div":
            return self.appr_div
        elif self.func is "exp":
            return self.appr_exp
        elif self.func is "log":
            return self.appr_log
        else:
            raise ValueError("Input function \'" + self.func + "\' is not supported in Appr_Func Class.")
            
            