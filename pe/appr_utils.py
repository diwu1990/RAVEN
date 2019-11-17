import torch
import matplotlib.pyplot as plt

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
            raise ValueError("Input rounding mode is not supported.")
    
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
    
    
def data_gen(data_range="0.0_1.0", mu=0.5, sigma = 0.3):
    # This is a function to generate data
    if data_range == "0.0_1.0":
        mu_eff = mu
    elif data_range == "0.5_1.0":
        mu_eff =  (mu - 0.5) * 2
        
    mu_tensor = torch.ones([100000]).mul(mu_eff)
    sigma_tensor = torch.ones([100000]).mul(sigma)
    data = torch.distributions.normal.Normal(mu_tensor, sigma_tensor).sample()
    data = data - data.floor()
    
    if data_range == "0.0_1.0":
        data = data
    elif data_range == "0.5_1.0":
        data =  data/2 + 0.5
    
    fig = plt.hist(data.cpu().numpy(), bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram for data")
    plt.show()
    
    return data

