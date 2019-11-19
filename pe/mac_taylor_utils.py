import torch
from RAVEN.pe.appr_utils import *

def MAC_Taylor(scale, 
               coeff, 
               var, 
               fxp=True, 
               intwidth=7, 
               fracwidth=8, 
               rounding_coeff="round", 
               rounding_var="round", 
               keepwidth=True):
    """
    Calculate the result of approximate Taylor series.
    The results is calculated as:
    output = scale * (const + sum(coeff * var^power))
    
    All inputs are tensors, except coeff, which is a list.
    
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
                 rounding_coeff="round", 
                 rounding_var="round", 
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
        coeff = [Trunc_val(i, intwidth = 2 * intwidth + 1, fracwidth = 2 * fracwidth, rounding = rounding_coeff) for i in coeff]
        coeff[-1] = Trunc_val(coeff[-1], intwidth = intwidth, fracwidth = fracwidth, rounding = rounding_coeff)
        
        # calculate the first term
        mul_0 = var
        mul_1 = coeff[-1]
        prod = mul_0.mul(mul_1)
        
        add_0 = coeff[-2]
        add_1 = prod
        acc = add_1.add(add_0)
        
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
            acc = add_1.add(add_0)
        
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
    
    
def point_search(func="exp", uniform=True, fxp=True, intwidth=7, fracwidth=8, valid=True, rounding_coeff="round", rounding_var="round", keepwidth=True):
    output_file = func + "_rms_error"
    if fxp is True:
        output_file = output_file + "_fxp_" + str(fracwidth)
    else:
        output_file = output_file + "_flp"
        
    if uniform is True:
        output_file = output_file + "_uniform"
    else:
        output_file = output_file + "_guassian"
        
    output_file = output_file + ".csv"
    
    f = open(output_file, "w+")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # choose data range according to function
    if func == "div":
        data_range = "0.5_1.0"
        point_list = [1.]
        mu_list = [i / 2 + 0.5 for i in [0.25, 0.5, 0.75]]
    if func == "exp":
        data_range = "0.0_1.0"
        # varying the Taylor expansion point
        # point_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        point_list = [0.00, 0.25, 0.50, 0.75, 1.00]
        # varying the distribution of data
        mu_list = [0.25, 0.5, 0.75]
    if func == "log":
        data_range = "0.5_1.0"
        # point_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.]
        point_list = [0.500, 0.625, 0.750, 0.875, 1.00]
        mu_list = [i / 2 + 0.5 for i in [0.25, 0.5, 0.75]]
    
    if uniform == True:
        sigma = 2
        mu_list = [0.5]
        f.write("uniform, \n")
    elif uniform == False:
        sigma = 0.2
        f.write("guassian, \n")
    
    for mu_value in mu_list:
        f.write("mu, "+str(mu_value)+",\n")

        data = data_gen(data_range=data_range, mu=mu_value, sigma=sigma).to(device)
        if func == "log":
            data = Trunc(data, 
                         intwidth=intwidth, 
                         fracwidth=fracwidth, 
                         rounding="floor")
        else:
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
            f.write("point, "+str(point)+",\n")
            
            coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
            if func == "exp":
                coeff = [1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]
            elif func == "div":
                coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
            elif func == "log":
                coeff[0] = 0 - math.log(point)
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
                    temp_scale = torch.exp(torch.tensor([point])).to(device)
                    temp_var = data - point
                elif func == "div":
                    temp_scale = torch.tensor([point]).to(device)
                    temp_var = point - data
                elif func == "log":
                    temp_scale = torch.tensor([-1.]).to(device)
                    temp_var = point - data
                
                appr_result = MAC_Taylor(temp_scale, 
                                         temp_coeff, 
                                         temp_var, 
                                         fxp=fxp, 
                                         intwidth=intwidth, 
                                         fracwidth=fracwidth, 
                                         rounding_coeff=rounding_coeff, 
                                         rounding_var=rounding_var, 
                                         keepwidth=keepwidth)
                error = (appr_result - ref_result) / ref_result
                min_err.append(error.min())
                max_err.append(error.max())
                avg_err.append(error.mean())
                rms_err.append(error.mul(error).mean().sqrt())
            
            eff_coeff = coeff[:]
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
            
            for i in rms_err:
                f.write(str(i.item())+", ")
            
            f.write("\n")
            
            print("eff coeff:", ["{0:0.10f}".format(i) for i in eff_coeff])
            print("min error:", ["{0:0.10f}".format(i) for i in min_err])
            print("max error:", ["{0:0.10f}".format(i) for i in max_err])
            print("avg error:", ["{0:0.10f}".format(i) for i in avg_err])
            print("rms error:", ["{0:0.10f}".format(i) for i in rms_err])
            print("")
            
        f.write("\n")
