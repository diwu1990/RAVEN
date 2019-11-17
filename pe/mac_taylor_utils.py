import torch
from RAVEN.pe.appr_utils import *

def MAC_Taylor(scale, 
               coeff, 
               var, 
               fxp=True, 
               intwidth=7, 
               fracwidth=8, 
               rounding_coeff="ceil", 
               rounding_var="floor", 
               keepwidth=True):
    """
    Calculate the result of approximate Taylor series.
    The results is calculated as:
    output = scale * (const + sum(coeff * var^power))
    
    All inputs are tensors.
    
    Assume that there are N taylor terms, which is the length of coeff, power always goes from 0 to N-1.
    1) Tensor scale is calculated by shifting input or using very small LUT.
    2) Tensor var is calulated as (input - offset)
    
    "fxp" means whether to performance fixed point calculation, in which the data bitwidth can be expressed as
    (1 + "intwidth" + "fracwidth"), and "rounding" indicates the rounding mode.
    """
    
    def flp_poly(scale, 
                 coeff, 
                 var):
        # initialization
        acc = torch.zeros_like(var)
        
        # cumulate the terms
        for idx in range(0, len(coeff)):
            acc.add_(coeff[idx] * var.pow(idx))
            
        return acc.mul(scale)
    
    def fxp_poly(scale, 
                 coeff, 
                 var, 
                 intwidth=7, 
                 fracwidth=8, 
                 rounding_coeff="ceil", 
                 rounding_var="floor", 
                 keepwidth=True):
        # 1) For multiplication,
        # each input has the format of (1, intwidth, fracwidth),
        # the output has the format of (1, 2 * intwidth + 1, 2 * fracwidth)
        # 2) For accumulation,
        # the input has the format of (1, 2 * intwidth + 1, 2 * fracwidth)
        # the output has the format of (1, 2 * intwidth + 2, 2 * fracwidth)

        acc = torch.zeros_like(var)

        # scale and var have format of (1, intwidth, fracwidth), as they both participate 
        # in the multiplication.
        scale = Trunc(scale, 
                      intwidth = intwidth, 
                      fracwidth = fracwidth, 
                      rounding = rounding_var)
        
        var = Trunc(var, 
                    intwidth = intwidth, 
                    fracwidth = fracwidth, 
                    rounding = rounding_var)

        # coeff participates in accumulation, so it has format of (1, 2 * intwidth + 1, 2 * fracwidth)
        coeff = Trunc(coeff, 
                      intwidth = 2 * intwidth + 1, 
                      fracwidth = 2 * fracwidth, 
                      rounding = rounding_coeff)
        
        # calculate the first term
        mul_0 = var
        mul_1 = Trunc(coeff[-1], 
                      intwidth = intwidth, 
                      fracwidth = fracwidth, 
                      rounding = rounding_coeff)
        prod = mul_0.mul(mul_1)
        
        add_0 = coeff[-2]
        add_1 = prod
        acc = add_0.add(add_1)
        
        # cumulate the rest terms
        for idx in range(0, len(coeff)-2):
            mul_0 = var
            mul_1 = Trunc(acc, 
                          intwidth = intwidth, 
                          fracwidth = fracwidth, 
                          rounding = rounding_var)
            prod = mul_0.mul(mul_1)

            add_0 = coeff[-3 - idx]
            add_1 = prod
            acc = add_0.add(add_1)
        
        mul_0 = scale
        mul_1 = Trunc(acc, 
                      intwidth = intwidth, 
                      fracwidth = fracwidth, 
                      rounding = rounding_var)
        prod = mul_0.mul(mul_1)
        
        output = prod
        
        if keepwidth is True:
            # output has the same bitwidth as input var
            return Trunc(output, 
                          intwidth = intwidth, 
                          fracwidth = fracwidth, 
                          rounding = rounding_var)
        elif keepwidth is False:
            # maintain the bitwidth of multiplier
            # bitwidth of output is twice that of input var
            return output
        else:
            raise ValueError("Input keepwidth mode need to be of bool type.")
    
    if fxp is True:
        return fxp_poly(scale, 
                        coeff, 
                        var, 
                        intwidth=intwidth, 
                        fracwidth=fracwidth, 
                        rounding_coeff=rounding_coeff, 
                        rounding_var=rounding_var, 
                        keepwidth=keepwidth)
    elif fxp is False:
        return flp_poly(scale, 
                        coeff, 
                        var)
    else:
        raise ValueError("Input fxp mode have to be of bool type.")
    
    
def point_search(func="exp", uniform=True, intwidth=7, fracwidth=8, valid=True, rounding_coeff="round", rounding_var="round"):
    # choose data range according to function
    if func == "div":
        data_range = "0.5_1.0"
        point_list = [1.]
        mu_list = [i / 2 + 0.5 for i in [0.25, 0.5, 0.75]]
    if func == "exp":
        data_range = "0.0_1.0"
        # varying the Taylor expansion point
        point_list = [0.0, 0.25, 0.500, 0.750, 1.]
        # varying the distribution of data
        mu_list = [0.25, 0.5, 0.75]
    if func == "log":
        data_range = "0.5_1.0"
        point_list = [0.500, 0.625, 0.750, 0.875, 1.]
        mu_list = [i / 2 + 0.5 for i in [0.25, 0.5, 0.75]]
    
    if uniform == True:
        sigma = 2
        mu_list = [0.5]
    elif uniform == False:
        sigma = 0.2
        
    for mu_value in mu_list:
        data = data_gen(data_range=data_range, mu=mu_value, sigma=sigma).cuda()
        data = Trunc(data, 
                     intwidth=intwidth, 
                     fracwidth=fracwidth, 
                     rounding=rounding_var)
        if func == "div":
            ref_result = torch.div(1, data)
        if func == "exp":
            ref_result = torch.exp(data)
        if func == "log":
            ref_result = torch.log(data)
        
        for point in point_list:
            coeff = torch.zeros(10).cuda()
            if func == "exp":
                coeff = torch.tensor([1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]).cuda()
            elif func == "div":
                coeff = torch.tensor([1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]).cuda()
            elif func == "log":
                coeff[0] = 0 - torch.log(torch.tensor([point])).item()
                for idx in range(1, 9):
                    coeff[idx] = 1/(point**idx)/idx
                    
            min_err = []
            max_err = []
            avg_err = []
            rms_err = []
            print("gaussian data mu=", mu_value, "Taylor expansion point=", point)

            for idx in range(2, len(coeff)):
                temp_coeff = coeff[0:idx]
                if func == "exp":
                    temp_scale = torch.exp(torch.tensor([point])).cuda()
                    temp_var = data - point
                elif func == "div":
                    temp_scale = torch.tensor([point]).cuda()
                    temp_var = point - data
                elif func == "log":
                    temp_scale = torch.tensor([-1.]).cuda()
                    temp_var = point - data
                
                appr_result = MAC_Taylor(temp_scale, 
                                         temp_coeff, 
                                         temp_var, 
                                         fxp=True, 
                                         intwidth=intwidth, 
                                         fracwidth=fracwidth, 
                                         rounding_coeff=rounding_coeff, 
                                         rounding_var=rounding_var, 
                                         keepwidth=True)
                error = (appr_result - ref_result) / ref_result
                min_err.append(error.min())
                max_err.append(error.max())
                avg_err.append(error.mean())
                rms_err.append(error.mul(error).mean().sqrt())
            
            # final check for useless round
            if valid is True:
                final_rms_err = rms_err[-1]
                valid_length = len(rms_err)
                temp_valid_length = len(rms_err)

                for term_idx in range(1, len(rms_err)):
                    if final_rms_err == rms_err[len(rms_err) - 1 - term_idx]:
                        temp_valid_length = len(rms_err) - term_idx
                        break
                valid_length = temp_valid_length

                least_rms_err = rms_err[valid_length-1]
                for term_idx in range(1, valid_length):
                    if least_rms_err >= rms_err[valid_length - 1 - term_idx]:
                        least_rms_err = rms_err[valid_length - 1 - term_idx]
                        temp_valid_length = valid_length - term_idx
                valid_length = temp_valid_length

                eff_coeff = coeff[0:valid_length]
                min_err = min_err[0:valid_length]
                max_err = max_err[0:valid_length]
                avg_err = avg_err[0:valid_length]
                rms_err = rms_err[0:valid_length]
                
            print("eff coeff:", ["{0:0.7f}".format(i) for i in eff_coeff])
            print("min error:", ["{0:0.7f}".format(i) for i in min_err])
            print("max error:", ["{0:0.7f}".format(i) for i in max_err])
            print("avg error:", ["{0:0.7f}".format(i) for i in avg_err])
            print("rms error:", ["{0:0.7f}".format(i) for i in rms_err])
            print("")