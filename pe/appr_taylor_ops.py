import torch
from RAVEN.pe.appr_poly_utils import Appr_Taylor

class ApprExp(torch.autograd.Function):
    """
    ApprExp is the approximate exponentiation with the gradient for the 
    approximate exponentiation.
    The gradient is always floating-point, regardless of approximation or 
    not.
    A precise exponentiation example can be found here:
    https://pytorch.org/docs/stable/_modules/torch/autograd/function.html
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, 
                bitwidth=16, 
                distribution="uniform", 
                fxp=True, 
                rounding="floor", 
                keepwidth=True, 
                appr_grad=False, 
                fxp_grad=False):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(x)
        
        if bitwidth == 16:
            ctx.intwidth=7
            ctx.fracwidth=8
        elif bitwidth == 12:
            ctx.intwidth=5
            ctx.fracwidth=6
        elif bitwidth == 8:
            ctx.intwidth=3
            ctx.fracwidth=4
        else:
            raise ValueError("Input bitwidth is not supported yet.")
            
        # Here the length of three tuples (coeff, power, and sign) should be 
        # the same
        if distribution == "uniform":
            # parameters for approximate polynomial in forward pass
            ctx.coeff_fw = coeff_fw
            ctx.power_fw = power_fw
            ctx.sign_fw  = sign_fw
            
            # parameters for approximate polynomial in backward pass
            ctx.coeff_bw = coeff_bw
            ctx.power_bw = power_bw
            ctx.sign_bw  = sign_bw
        else:
            raise ValueError("Input distribution is not supported yet.")
        
        ctx.fxp = fxp
        ctx.rounding = rounding
        ctx.keepwidth = keepwidth
        ctx.appr_grad = appr_grad
        ctx.fxp_grad = fxp_grad
        
        scale = scale
        const = const
        var = x
        output = Appr_Taylor(scale, 
                             const, 
                             var, 
                             ctx.coeff_fw, 
                             ctx.power_fw, 
                             ctx.sign_fw, 
                             fxp=ctx.fxp, 
                             intwidth=ctx.intwidth, 
                             fracwidth=ctx.fracwidth, 
                             rounding=ctx.rounding, 
                             keepwidth=ctx.keepwidth)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        x, = ctx.saved_tensors
        grad_input = None

        # Calculate the gradient according to the flag
        if ctx.appr_grad is True:
            grad = Appr_Taylor(scale, 
                               const, 
                               var, 
                               ctx.coeff_bw, 
                               ctx.power_bw, 
                               ctx.sign_bw, 
                               fxp=ctx.fxp_grad, 
                               intwidth=ctx.intwidth, 
                               fracwidth=ctx.fracwidth, 
                               rounding=ctx.rounding, 
                               keepwidth=ctx.keepwidth)
            grad_input = grad_output * grad
        elif ctx.appr_grad is False:
            grad_input = grad_output * torch.exp(x)
        else:
            raise ValueError("Input appr_grad need to be of bool type.")
            
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_input, None, None, None, None, None, None, None

    
