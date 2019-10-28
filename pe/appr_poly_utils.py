import torch

def Poly_Calc(x, coeff, power, sign):
    """
    Calculate the result for polynomial.
    """
    # check the size of coeff, power and sign are the same.
    len_coeff = coeff.size()
    len_power = power.size()
    len_sign  = sign.size()
    assert (len_coeff == len_power and len_coeff == len_sign), "Non-indentical size for coeff, power and sign."
    
    output = torch.ones_like(x)
    for idx in range(len_coeff):
        output += coeff * input**power * sign
    
    
    
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
            
            