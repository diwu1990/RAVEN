import torch
from RAVEN.pe.appr_utils import RoundingNoGrad, Trunc

# This file is for SECO.

def Appr_Taylor(scale, 
                const, 
                var, 
                coeff, 
                power, 
                sign, 
                fxp=True, 
                intwidth=7, 
                fracwidth=8, 
                rounding="floor", 
                keepwidth=True):
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
    
    def flp_poly(scale, 
                 const, 
                 var, 
                 coeff, 
                 power, 
                 sign):
        # initialization
        acc = torch.zeros_like(var).add(const)
        
        # cumulate the terms
        for idx in range(0, len(coeff)):
            acc.add_(coeff[idx] * var.pow(power[idx]) * sign[idx])
            
        return acc.mul(scale)
    
    def fxp_poly(scale, 
                 const, 
                 var, 
                 coeff, 
                 power, 
                 sign, 
                 intwidth=7, 
                 fracwidth=8, 
                 rounding="floor", 
                 keepwidth=True):
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
        
        # both scale and var are of format (1, intwidth, fracwidth), as 
        # they participate in the multiplication
        scale = Trunc(scale, 
                      intwidth = intwidth, 
                      fracwidth = fracwidth, 
                      rounding = rounding)
        
        var = Trunc(var, 
                    intwidth = intwidth, 
                    fracwidth = fracwidth, 
                    rounding = rounding)

        # const is of format (1, 2 * intwidth + 1, 2 * fracwidth), as it 
        # participates in the accumulation
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
            raise ValueError("Input keepwidth mode need to be of bool type.")
    
    if fxp is True:
        return fxp_poly(scale, 
                        const, 
                        var, 
                        coeff, 
                        power, 
                        sign, 
                        intwidth=intwidth, 
                        fracwidth=fracwidth, 
                        rounding=rounding, 
                        keepwidth=keepwidth)
    elif fxp is False:
        return flp_poly(scale, 
                        const, 
                        var, 
                        coeff, 
                        power, 
                        sign)
    else:
        raise ValueError("Input fxp mode have to be of bool type.")
        

def param_search(max_extra_term, 
                 max_shift, 
                 max_power_diff, 
                 max_sign_change, 
                 ref_result, 
                 point, 
                 scale, 
                 const, 
                 var, 
                 coeff, 
                 power, 
                 sign, 
                 fxp=True, 
                 intwidth=7, 
                 fracwidth=8, 
                 rounding="floor", 
                 keepwidth=True, 
                 valid=True):
    """
    search parameters for approximate Taylor series with discrete gradient descend algorithm
    """
    
    min_err = []
    max_err = []
    avg_err = []
    rms_err = []
    
    for term_idx in range(len(coeff)):
        appr_result = Appr_Taylor(scale,
                                  const, 
                                  var, 
                                  coeff[0:term_idx+1], 
                                  power[0:term_idx+1], 
                                  sign[0:term_idx+1], 
                                  fxp=fxp, 
                                  intwidth=intwidth, 
                                  fracwidth=fracwidth, 
                                  rounding=rounding, 
                                  keepwidth=keepwidth)
        appr_err = (appr_result - ref_result)/ref_result

        min_err.append(appr_err.min().item())
        max_err.append(appr_err.max().item())
        avg_err.append(appr_err.mean().item())
        rms_err.append(appr_err.mul(appr_err).mean().sqrt().item())

    # this loop is for number of terms
    for term_idx in range(max_extra_term):
        temp_coeff = coeff[:]
        temp_coeff.append(coeff[-1])
        temp_power = power[:]
        temp_power.append(power[-1])
        temp_sign  = sign[:]
        temp_sign.append(sign[-1])

        temp_min_error = 0
        temp_max_error = 0
        temp_avg_error = 0
        temp_rms_error = 10000000000

        # this loop is for shifting offset of coeff
        for coeff_idx in range(-max_shift, max_shift+1):
            temp_coeff[-1] = coeff[-1] * (2**coeff_idx)

            # this loop is for multiplication of var
            for power_idx in range(max_power_diff):
                temp_power[-1] = power[-1] + power_idx

                # this loop is for accumulation of terms
                for sign_idx in range(max_sign_change):
                    temp_sign[-1]  = sign[-1] * (1-2*sign_idx)

                    appr_result = Appr_Taylor(scale,
                                              const, 
                                              var, 
                                              temp_coeff, 
                                              temp_power, 
                                              temp_sign, 
                                              fxp=fxp, 
                                              intwidth=intwidth, 
                                              fracwidth=fracwidth, 
                                              rounding=rounding, 
                                              keepwidth=keepwidth)
                    appr_err = (appr_result - ref_result)/ref_result

                    if appr_err.mul(appr_err).mean().sqrt().item() < temp_rms_error:
                        new_coeff = temp_coeff[-1]
                        new_power = temp_power[-1]
                        new_sign  = temp_sign[-1]
                        temp_min_error = appr_err.min().item()
                        temp_max_error = appr_err.max().item()
                        temp_avg_error = appr_err.mean().item()
                        temp_rms_error = appr_err.mul(appr_err).mean().sqrt().item()
                        temp_all_error = appr_err

        coeff.append(new_coeff)
        power.append(new_power)
        sign.append(new_sign)
        min_err.append(temp_min_error)
        max_err.append(temp_max_error)
        avg_err.append(temp_avg_error)
        rms_err.append(temp_rms_error)

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
        
        coeff = coeff[0:valid_length]
        power = power[0:valid_length]
        sign  = sign[0:valid_length]
        min_err = min_err[0:valid_length]
        max_err = max_err[0:valid_length]
        avg_err = avg_err[0:valid_length]
        rms_err = rms_err[0:valid_length]

    print("Approximate Taylor series at point:", point)
    print("final coeff", coeff)
    print("final power", power)
    print("final sign", sign)

    print("min error:", ["{0:0.5f}".format(i) for i in min_err])
    print("max error:", ["{0:0.5f}".format(i) for i in max_err])
    print("avg error:", ["{0:0.5f}".format(i) for i in avg_err])
    print("rms error:", ["{0:0.5f}".format(i) for i in rms_err])
    print("")
    
    return coeff, power, sign, min_err, max_err, avg_err, rms_err
    
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# the following codes are used to generate the parameters used for exp
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def exp_data_gen(distribution="uniform"):
    # This is a function to generate data for chosing parameter for exp, which
    # only cares about the data between 0 and 1, because data outside this can
    # be scaled using LUT.
    if distribution == "uniform":
        data = torch.distributions.uniform.Uniform(torch.zeros([100000]), torch.ones([100000])).sample()
    elif distribution == "middle":
        mu = torch.ones([100000]).mul(0.5)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = data - data.floor()
    elif distribution == "left":
        mu = torch.ones([100000]).mul(0.25)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = data - data.floor()
    elif distribution == "right":
        mu = torch.ones([100000]).mul(0.75)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = data - data.floor()
    else:
        raise ValueError("Input distribution mode is not supported.")
    return data


def exp_param_gen(distribution="uniform", intwidth=7, fracwidth=8, rounding="round", keepwidth=True, valid=True):
    # max number of extra terms besides the initial terms
    max_extra_term = 10
    # max shifting offset, including for both left and right shifting. In total, there will be 2*max_shift+1 cases
    max_shift = 3
    # max power difference, this value is fixed to 2: 0 means no mul is skipped this cycle, while 1 mean mul is done this cycle.
    max_power_diff = 2
    # max sign change, this value is fixed to 2: 0 means no change from last sign, while 1 means changing sign from last.
    max_sign_change = 2
    
    # define the floating-point input
    data = exp_data_gen(distribution)
    
    # reference model
    ref_result = torch.exp(data)

    # approximate taylor series
    point_list = [0.0, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875]

    for point in point_list:
        scale = torch.exp(torch.Tensor([point]))
        const = torch.Tensor([1.])
        var   = data - point
        
        coeff = [1/1, 1/2]
        power = [  1,   2]
        sign  = [  1,   1]
        
        coeff, power, sign, min_err, max_err, avg_err, rms_err = param_search(max_extra_term, 
                                                                              max_shift, 
                                                                              max_power_diff, 
                                                                              max_sign_change, 
                                                                              ref_result, 
                                                                              point, 
                                                                              scale, 
                                                                              const, 
                                                                              var, 
                                                                              coeff, 
                                                                              power, 
                                                                              sign, 
                                                                              fxp=True, 
                                                                              intwidth=intwidth, 
                                                                              fracwidth=fracwidth, 
                                                                              rounding=rounding, 
                                                                              keepwidth=keepwidth, 
                                                                              valid=valid)
        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# the following codes are used to generate the parameters used for div
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def div_data_gen(distribution="uniform"):
    # This is a function to generate data for chosing parameter for div(y, x), which
    # only cares about the x between 0.5 and 1. only scale is related to y.
    if distribution == "uniform":
        data = torch.distributions.uniform.Uniform(torch.zeros([100000])+0.5, torch.ones([100000])).sample()
    elif distribution == "middle":
        mu = torch.ones([100000]).mul(0.5)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = 1 - (data - data.floor())/2
    elif distribution == "right":
        mu = torch.ones([100000]).mul(0.25)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = 1 - (data - data.floor())/2
    elif distribution == "left":
        mu = torch.ones([100000]).mul(0.75)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = 1 - (data - data.floor())/2
    else:
        raise ValueError("Input distribution mode is not supported.")
    return data


def div_param_gen(distribution="uniform", intwidth=7, fracwidth=8, rounding="round", keepwidth=True, valid=True):
    # for div y/x, just need to approximate the 1/x with sum(-1*(x-1))^i
    
    # max number of extra terms besides the initial terms
    max_extra_term = 2
    # max shifting offset, including for both left and right shifting. In total, there will be 2*max_shift+1 cases
    max_shift = 3
    # max power difference, this value is fixed to 2: 0 means no mul is skipped this cycle, while 1 mean mul is done this cycle.
    max_power_diff = 2
    # max sign change, this value is fixed to 2: 0 means no change from last sign, while 1 means changing sign from last.
    max_sign_change = 2
    
    # define the floating-point input
    data = div_data_gen(distribution)
    
    # reference model
    ref_result = torch.div(1, data)

    # approximate taylor series
    point_list = [1.0]

    for point in point_list:
        scale = torch.Tensor([point])
        const = torch.Tensor([1.])
        var   = data - point
        
        coeff = [1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1]
        power = [  1,   2,   3,   4,   5,   6,   7,   8,   9]
        sign  = [ -1,   1,  -1,   1,  -1,   1,  -1,   1,  -1]
        
        coeff, power, sign, min_err, max_err, avg_err, rms_err = param_search(max_extra_term, 
                                                                              max_shift, 
                                                                              max_power_diff, 
                                                                              max_sign_change, 
                                                                              ref_result, 
                                                                              point, 
                                                                              scale, 
                                                                              const, 
                                                                              var, 
                                                                              coeff, 
                                                                              power, 
                                                                              sign, 
                                                                              fxp=True, 
                                                                              intwidth=intwidth, 
                                                                              fracwidth=fracwidth, 
                                                                              rounding=rounding, 
                                                                              keepwidth=keepwidth, 
                                                                              valid=valid)
        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# the following codes are used to generate the parameters used for log
# this method is not working well for log, acuracy is too low
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def log_data_gen(distribution="uniform"):
    # This is a function to generate data for chosing parameter for log, which
    # only cares about the data between 0 and 1, because log is only used after softmax.
    if distribution == "uniform":
        data = torch.distributions.uniform.Uniform(torch.zeros([100000])+0.5, torch.ones([100000])).sample()
    elif distribution == "middle":
        mu = torch.ones([100000]).mul(0.5)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = 1 - (data - data.floor())/2
    elif distribution == "right":
        mu = torch.ones([100000]).mul(0.25)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = 1 - (data - data.floor())/2
    elif distribution == "left":
        mu = torch.ones([100000]).mul(0.75)
        sigma = torch.ones([100000]).mul(0.3)
        data = torch.distributions.normal.Normal(mu, sigma).sample()
        data = 1 - (data - data.floor())/2
    else:
        raise ValueError("Input distribution mode is not supported.")
    return data


def log_param_gen(distribution="uniform", intwidth=7, fracwidth=8, rounding="round", keepwidth=True, valid=True):
    # max number of extra terms besides the initial terms
    max_extra_term = 1
    # max shifting offset, including for both left and right shifting. In total, there will be 2*max_shift+1 cases
    max_shift = 3
    # max power difference, this value is fixed to 2: 0 means no mul is skipped this cycle, while 1 mean mul is done this cycle.
    max_power_diff = 2
    # max sign change, this value is fixed to 2: 0 means no change from last sign, while 1 means changing sign from last.
    max_sign_change = 2
    
    # define the floating-point input
    data = log_data_gen(distribution)
    
    # reference model
    ref_result = torch.log(data)
    print(ref_result)

    # approximate taylor series
    point_list = [1.0]

    for point in point_list:
        scale = torch.Tensor([point])
        const = torch.log(torch.Tensor([point]))
        var   = data - point
        print(scale, const)
        coeff = [1/1, 1/2, 1/4, 1/4, 1/4, 1/8]
        power = [  1,   2,   3,   4,   5,   6]
        sign  = [  1,  -1,   1,  -1,   1,  -1]
        
        coeff, power, sign, min_err, max_err, avg_err, rms_err = param_search(max_extra_term, 
                                                                              max_shift, 
                                                                              max_power_diff, 
                                                                              max_sign_change, 
                                                                              ref_result, 
                                                                              point, 
                                                                              scale, 
                                                                              const, 
                                                                              var, 
                                                                              coeff, 
                                                                              power, 
                                                                              sign, 
                                                                              fxp=True, 
                                                                              intwidth=intwidth, 
                                                                              fracwidth=fracwidth, 
                                                                              rounding=rounding, 
                                                                              keepwidth=keepwidth, 
                                                                              valid=valid)
    
