import torch
import math
from RAVEN.pe.raven.utils import RAVENtaylor

# This file is for RAVEN PE simulation from "When Dataflows Converge: Reconfigurable and Approximate Computing for Emerging Neural Networks"


class RAVENexpFunc(torch.autograd.Function):
    """
    RAVENexpFunc is the approximate exponentiation exp(x) with gradient.
    A precise exponentiation example can be found here:
    https://pytorch.org/docs/stable/_modules/torch/autograd/function.html
    """
    @staticmethod
    def forward(ctx, x, offset, scale, var, cycle, coeff, intwidth, fracwidth, rounding):
        output = RAVENtaylor(offset, scale, var, cycle, coeff, intwidth, fracwidth, rounding)
        ctx.save_for_backward(x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output * torch.exp(x)
        return grad_x, None, None, None, None, None, None, None, None


class RAVENexp(torch.nn.Module):
    def __init__(self, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENexp, self).__init__()
        self.cycle = cycle
        self.rounding = rounding
        self.coeff = torch.nn.Parameter(torch.Tensor([1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880][0:cycle]), requires_grad=False)
        self.intwidth = torch.nn.Parameter(torch.ones(cycle) * intwidth_max, requires_grad=False)
        if bitwidth_reduce is True:
            self.fracwidth = torch.nn.Parameter(torch.nn.functional.relu(torch.arange(fracwidth_max, fracwidth_max - cycle, -1)), requires_grad=False)
        else:
            self.fracwidth = torch.nn.Parameter(torch.ones(cycle) * fracwidth_max, requires_grad=False)

    def forward(self, x):
        int_part = x.floor()
        frac_part = x - int_part
        offset = torch.zeros_like(x)
        scale = torch.exp(int_part)
        var = frac_part
        output = RAVENexpFunc.apply(x, offset, scale, var, self.cycle, self.coeff, self.intwidth, self.fracwidth, self.rounding)
        return output


class RAVENdivFunc(torch.autograd.Function):
    """
    RAVENdivFunc is the approximate division (y/x) with gradient.
    """
    @staticmethod
    def forward(ctx, y, x, offset, scale, var, cycle, coeff, intwidth, fracwidth, rounding):
        output = RAVENtaylor(offset, scale, var, cycle, coeff, intwidth, fracwidth, rounding)
        ctx.save_for_backward(y, x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        grad_y = grad_output * torch.div(1, x)
        grad_x = grad_output * (0 - torch.div(y, x.mul(x)))
        return grad_y, grad_x, None, None, None, None, None, None, None, None


class RAVENdiv(torch.nn.Module):
    def __init__(self, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENdiv, self).__init__()
        self.cycle = cycle
        self.rounding = rounding
        self.point = 0.75
        coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
        for idx in range(0, 10):
            coeff[idx] = 1/(self.point**(idx+1))
        self.coeff = torch.nn.Parameter(torch.Tensor(coeff[0:cycle]), requires_grad=False)
        self.intwidth = torch.nn.Parameter(torch.ones(cycle) * intwidth_max, requires_grad=False)
        if bitwidth_reduce is True:
            self.fracwidth = torch.nn.Parameter(torch.nn.functional.relu(torch.arange(fracwidth_max, fracwidth_max - cycle, -1)), requires_grad=False)
        else:
            self.fracwidth = torch.nn.Parameter(torch.ones(cycle) * fracwidth_max, requires_grad=False)

    def forward(self, y, x):
        shift = torch.log2(x).floor() + 1
        offset = torch.zeros_like(x)
        scale = y >> shift
        var = self.point - (x >> shift)
        output = RAVENdivFunc.apply(y, x, offset, scale, var, self.cycle, self.coeff, self.intwidth, self.fracwidth, self.rounding)
        return output


class RAVENlogFunc(torch.autograd.Function):
    """
    RAVENlogFunc is the approximate logarithm log(x) with gradient.
    """
    @staticmethod
    def forward(ctx, x, offset, scale, var, cycle, coeff, intwidth, fracwidth, rounding):
        output = RAVENtaylor(offset, scale, var, cycle, coeff, intwidth, fracwidth, rounding)
        ctx.save_for_backward(x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output * torch.div(1, x)
        return grad_x, None, None, None, None, None, None, None, None


class RAVENlog(torch.nn.Module):
    def __init__(self, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENlog, self).__init__()
        self.cycle = cycle
        self.rounding = rounding
        self.point = 0.75
        coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
        coeff[0] = 0 - math.log(self.point)
        for idx in range(1, 9):
            coeff[idx] = 1/(self.point**idx)/idx
        self.coeff = torch.nn.Parameter(torch.Tensor(coeff[0:cycle]), requires_grad=False)
        self.intwidth = torch.nn.Parameter(torch.ones(cycle) * intwidth_max, requires_grad=False)
        if bitwidth_reduce is True:
            self.fracwidth = torch.nn.Parameter(torch.nn.functional.relu(torch.arange(fracwidth_max, fracwidth_max - cycle, -1)), requires_grad=False)
        else:
            self.fracwidth = torch.nn.Parameter(torch.ones(cycle) * fracwidth_max, requires_grad=False)

    def forward(self, x):
        shift = torch.log2(x).floor() + 1
        offset = shift * math.log(2.)
        scale = 0 - torch.ones_like(x)
        var = self.point - (x >> shift)
        output = RAVENlogFunc.apply(x, offset, scale, var, self.cycle, self.coeff, self.intwidth, self.fracwidth, self.rounding)
        return output


class RAVENsigmoid(torch.nn.Module):
    def __init__(self, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENsigmoid, self).__init__()
        self.exp = RAVENexp(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)
        self.div = RAVENdiv(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)

    def forward(self, x):
        exp_val = self.exp(-x)
        div_val = self.div(torch.ones_like(exp_val), 1. + exp_val)
        return div_val


class RAVENtanh(torch.nn.Module):
    def __init__(self, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENtanh, self).__init__()
        self.exp = RAVENexp(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)
        self.div = RAVENdiv(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)

    def forward(self, x):
        exp_val = self.exp(2*x)
        div_val = 1 - self.div(torch.ones_like(exp_val) << 1, exp_val + 1.)
        return div_val


class RAVENsoftmax(torch.nn.Module):
    def __init__(self, 
                 dim=0, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENsoftmax, self).__init__()
        self.dim = dim
        self.exp = RAVENexp(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)
        self.div = RAVENdiv(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)

    def forward(self, x):
        exp_val = self.exp(x)
        inf_check = torch.eq(exp_val, float('inf')).type(torch.float)
        inf_check_sum = inf_check.sum()
        if inf_check_sum.item() >= 1:
            return inf_check.mul(1/inf_check_sum)

        sum_val = exp_val.sum(dim=self.dim, keepdim=True)
        div_val = self.div(sum_val, exp_val)
        output = self.div(torch.ones_like(x), div_val)
        return output
    

class RAVENlogsoftmax(torch.nn.Module):
    def __init__(self, 
                 dim=0, 
                 cycle=4, 
                 intwidth_max=7, 
                 fracwidth_max=8, 
                 bitwidth_reduce=True, 
                 rounding="round"):
        super(RAVENlogsoftmax, self).__init__()
        self.dim = dim
        self.exp = RAVENexp(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)
        self.log = RAVENlog(cycle, intwidth_max, fracwidth_max, bitwidth_reduce, rounding)

    def forward(self, x):
        exp_val = self.exp(x)
        sum_val = exp_val.sum(dim=self.dim, keepdim=True)
        log_exp_val = self.log(exp_val)
        log_sum_val = self.log(sum_val)
        output = log_exp_val - log_sum_val
        return output

