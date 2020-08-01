import torch
import math
from RAVEN.pe.mac_taylor_utils import MAC_Taylor

# This file is for UNO PE.

class MACexp(torch.autograd.Function):
    """
    MACexp is the approximate exponentiation with the gradient for the 
    approximate exponentiation.
    The gradient is always floating-point, regardless of approximation or 
    not.
    A precise exponentiation example can be found here:
    https://pytorch.org/docs/stable/_modules/torch/autograd/function.html
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, 
                cycle, 
                intwidth=7, 
                fracwidth=8, 
                # distribution="uniform", # this param is currently not used
                fxp=True, 
                rounding="round", 
                keepwidth=False, 
                appr_grad=False, 
                fxp_grad=False):
        
        # coeff is the precise coefficient, up to 10 terms are pre-stored
        ctx.coeff = [1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]
        ctx.cycle = cycle - 1
        
        ctx.intwidth = intwidth
        ctx.fracwidth = fracwidth
        
        # parameters for approximate polynomial in forward and backward pass
        # both are zero because after changing floating to fixed point, the 
        # center of data will not have so much influence. They are not used.
        # ctx.point_fw = 0
        # ctx.point_bw = 0
        
        ctx.fxp = fxp
        ctx.rounding = rounding
        ctx.keepwidth = keepwidth
        ctx.appr_grad = appr_grad
        ctx.fxp_grad = fxp_grad
        
        int_part = x.floor()
        frac_part = x - int_part
        scale = torch.exp(int_part)
        var = frac_part
        output = MAC_Taylor(scale, 
                            ctx.coeff[0:ctx.cycle], 
                            var, 
                            fxp=ctx.fxp, 
                            intwidth=ctx.intwidth, 
                            fracwidth=ctx.fracwidth, 
                            rounding_coeff="round", 
                            rounding_var=ctx.rounding, 
                            keepwidth=True)
        
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        x, = ctx.saved_tensors
        grad_x = None

        # Calculate the gradient according to the flag
        if ctx.appr_grad is True:
            int_part = x.floor()
            frac_part = x - int_part
            scale = torch.exp(int_part)
            var = frac_part
            grad = MAC_Taylor(scale, 
                              ctx.coeff[0:ctx.cycle], 
                              var, 
                              fxp=ctx.fxp_grad, 
                              intwidth=ctx.intwidth, 
                              fracwidth=ctx.fracwidth, 
                              rounding_coeff="round", 
                              rounding_var=ctx.rounding, 
                              keepwidth=True)
            grad_x = grad_output * grad
        elif ctx.appr_grad is False:
            grad_x = grad_output * torch.exp(x)
        else:
            raise ValueError("Input appr_grad need to be of bool type.")
            
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_x, None, None, None, None, None, None, None, None

    
class MACdiv(torch.autograd.Function):
    """
    MACdiv is the approximate exponentiation with the gradient for the 
    approximate division (y/x).
    The gradient is always floating-point, regardless of approximation or 
    not.
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, y, x, 
                cycle, 
                intwidth=7, 
                fracwidth=8, 
                # distribution="uniform", # this param is currently not used
                fxp=True, 
                rounding="round", 
                keepwidth=False, 
                appr_grad=False, 
                fxp_grad=False):
        
        # coeff is the precise coefficient, up to 10 terms are pre-stored
        coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
        point = 0.75
        for idx in range(0, 10):
            coeff[idx] = 1/(point**(idx+1))
            
        ctx.coeff = coeff
        ctx.cycle = cycle - 1
        
        ctx.intwidth = intwidth
        ctx.fracwidth = fracwidth
        
        # parameters for approximate polynomial in forward and backward pass
        # both are zero because after changing floating to fixed point, the 
        # center of data will not have so much influence. They are not used.
        # ctx.point_fw = 0
        # ctx.point_bw = 0
        
        ctx.fxp = fxp
        ctx.rounding = rounding
        ctx.keepwidth = keepwidth
        ctx.appr_grad = appr_grad
        ctx.fxp_grad = fxp_grad
        
        scale = y >> (torch.log2(x).floor() + 1)
        var = point - (x >> (torch.log2(x).floor() + 1))
        output = MAC_Taylor(scale, 
                            ctx.coeff[0:ctx.cycle], 
                            var, 
                            fxp=ctx.fxp, 
                            intwidth=ctx.intwidth, 
                            fracwidth=ctx.fracwidth, 
                            rounding_coeff="round", 
                            rounding_var=ctx.rounding, 
                            keepwidth=True)
        
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(y, x)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        y, x = ctx.saved_tensors
        grad_y = None
        grad_x = None

        # Calculate the gradient according to the flag
        grad_y = grad_output * torch.div(1, x)
        grad_x = grad_output * (0 - torch.div(y, x.mul(x)))
            
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_y, grad_x, None, None, None, None, None, None, None, None


class MAClog(torch.autograd.Function):
    """
    MAClog is the approximate logarithm with the gradient for the 
    approximate.
    The gradient is always floating-point, regardless of approximation or 
    not.
    A precise exponentiation example can be found here:
    https://pytorch.org/docs/stable/_modules/torch/autograd/function.html
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, 
                cycle, 
                intwidth=7, 
                fracwidth=8, 
                # distribution="uniform", # this param is currently not used
                fxp=True, 
                rounding="round", 
                keepwidth=False, 
                appr_grad=False, 
                fxp_grad=False):
        
        # coeff is the precise coefficient for exp, up to 10 terms are pre-stored
        coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
        point = 0.75
        coeff[0] = 0 - math.log(point)
        for idx in range(1, 9):
            coeff[idx] = 1/(point**idx)/idx
            
        ctx.coeff = coeff
        ctx.cycle = cycle - 1
        
        ctx.intwidth = intwidth
        ctx.fracwidth = fracwidth
        
        # parameters for approximate polynomial in forward and backward pass
        # both are zero because after changing floating to fixed point, the 
        # center of data will not have so much influence. They are not used.
        # ctx.point_fw = 0
        # ctx.point_bw = 0
        
        ctx.fxp = fxp
        ctx.rounding = rounding
        ctx.keepwidth = keepwidth
        ctx.appr_grad = appr_grad
        ctx.fxp_grad = fxp_grad
        
        scale = 0 - torch.ones_like(x)
        var = point - (x >> (torch.log2(x).floor() + 1))
        output = MAC_Taylor(scale, 
                            ctx.coeff[0:ctx.cycle], 
                            var, 
                            fxp=ctx.fxp, 
                            intwidth=ctx.intwidth, 
                            fracwidth=ctx.fracwidth, 
                            rounding_coeff="round", 
                            rounding_var=ctx.rounding, 
                            keepwidth=True)
        
        output = output + (torch.log2(x).floor() + 1) * math.log(2.)
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(x)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        x, = ctx.saved_tensors
        grad_x = None

        # Calculate the gradient according to the flag
        grad_x = grad_output * torch.div(1, x)
        
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_x, None, None, None, None, None, None, None, None


def MACsigmoid(x, 
               cycle=8, 
               intwidth=7, 
               fracwidth=8, 
               fxp=True, 
               rounding="round", 
               keepwidth=False, 
               appr_grad=False, 
               fxp_grad=False):
    exp_val = MACexp.apply(-x, 
                           cycle, 
                           intwidth, 
                           fracwidth, 
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    div_val = MACdiv.apply(torch.ones_like(exp_val), 
                           1. + exp_val, 
                           cycle, 
                           intwidth, 
                           fracwidth, 
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    return div_val


def MACtanh(x, 
            cycle=8, 
            intwidth=7, 
            fracwidth=8, 
            fxp=True, 
            rounding="round", 
            keepwidth=False, 
            appr_grad=False, 
            fxp_grad=False):
    exp_val = MACexp.apply(2*x, 
                           cycle, 
                           intwidth, 
                           fracwidth, 
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    div_val = MACdiv.apply(exp_val - 1., 
                           exp_val + 1., 
                           cycle, 
                           intwidth, 
                           fracwidth, 
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    return div_val


def MACsoftmax(x, 
               dim=0, 
               cycle=8, 
               intwidth=7, 
               fracwidth=8, 
               fxp=True, 
               rounding="round", 
               keepwidth=False, 
               appr_grad=False, 
               fxp_grad=False):
    # when the input is too small, one solution from software side is too convert it to larger values.
    # but this might not solve the div-by-0 issule.
    # offset = torch.mean(x, dim=dim)
    # offset = torch.unsqueeze(offset, dim)
    # exp_val = MACexp.apply(x - offset, 
    exp_val = MACexp.apply(x, 
                           cycle, 
                           intwidth, 
                           fracwidth, 
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    inf_check = torch.eq(exp_val, float('inf')).type(torch.float)
    inf_check_sum = inf_check.sum()
    if inf_check_sum.item() >= 1:
        return inf_check.mul(1/inf_check_sum)
    sum_val = exp_val.sum(dim=dim, keepdim=True)

    div_val = MACdiv.apply(exp_val, 
                           sum_val, 
                           cycle, 
                           intwidth,
                           fracwidth,
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    return div_val


def MAClogsoftmax(x, 
                  dim=0, 
                  cycle=8, 
                  intwidth=7, 
                  fracwidth=8, 
                  fxp=True, 
                  rounding="round", 
                  keepwidth=False, 
                  appr_grad=False, 
                  fxp_grad=False):
    softmax_val = MACsoftmax(x, 
                             dim, 
                             cycle, 
                             intwidth, 
                             fracwidth, 
                             fxp, 
                             rounding, 
                             keepwidth, 
                             appr_grad, 
                             fxp_grad)
    log_val = MAClog.apply(softmax_val, 
                           cycle, 
                           intwidth, 
                           fracwidth, 
                           fxp, 
                           rounding, 
                           keepwidth, 
                           appr_grad, 
                           fxp_grad)
    log_val[torch.isnan(log_val)] = -2**intwidth
    return log_val

