# %%
import torch
from RAVEN.pe.raven.ops import RAVENexp, RAVENdiv, RAVENlog, RAVENsigmoid, RAVENtanh, RAVENsoftmax, RAVENlogsoftmax
from RAVEN.pe.uno.ops import UNOsigmoid, UNOtanh, UNOsoftmax, UNOlogsoftmax
import time
import argparse

# parameters for raven design
parser = argparse.ArgumentParser(description='RAVEN PE')
parser.add_argument('--cycle', type=int, default=8, metavar='C',
                    help='cycle count for nonlinear operation')
parser.add_argument('--intwidth-max', type=int, default=7, metavar='I',
                    help='maximum integer width')
parser.add_argument('--fracwidth-max', type=int, default=8, metavar='F',
                    help='maximum fracwidth width')
parser.add_argument('--bitwidth-reduce', action='store_true', default=False,
                    help='allows to reduce MAC bitwidth')
parser.add_argument('--rounding', type=str, default='round', metavar='R',
                    help='rounding mode')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='evaluate complex functions beyond div/exp/log')

global args
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cycle=args.cycle
intwidth_max=args.intwidth_max
fracwidth_max=args.fracwidth_max
bitwidth_reduce=args.bitwidth_reduce
rounding=args.rounding

####################################################################################
print("# # # # # # # # # # # # # # # #")
print("# Test RAVENdiv")
print("# # # # # # # # # # # # # # # #")
start, end, interval = 0.5, 1., 0.001
print("input range: ", start, end)

y = torch.tensor([1.]).to(device)
x = torch.arange(start, end, interval).to(device)
y.requires_grad_()
x.requires_grad_()
approximate = RAVENdiv(cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(y, x)
approximate.sum().backward()

y = torch.tensor([1.]).to(device)
x = torch.arange(start, end, interval).to(device)
y.requires_grad_()
x.requires_grad_()
precise = torch.div(y, x)
precise.sum().backward()

error = (approximate - precise) / precise
print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

error = (approximate - precise)
print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
print("\n")



####################################################################################
print("# # # # # # # # # # # # # # # #")
print("# Test RAVENexp")
print("# # # # # # # # # # # # # # # #")
start, end, interval = 0., 1., 0.001
print("input range: ", start, end)
x = torch.arange(start, end, interval).to(device)
x.requires_grad_()
approximate = RAVENexp(cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(x)
approximate.sum().backward()

x = torch.arange(start, end, interval).to(device)
x.requires_grad_()
precise = torch.exp(x)
precise.sum().backward()

error = (approximate - precise) / precise
print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

error = (approximate - precise)
print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
print("\n")



####################################################################################
print("# # # # # # # # # # # # # # # #")
print("# Test RAVENlog")
print("# # # # # # # # # # # # # # # #")
start, end, interval = 0.5, 1., 0.001
print("input range: ", start, end)
x = torch.arange(start, end, interval).to(device)
x.requires_grad_()
approximate = RAVENlog(cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(x)
approximate.sum().backward()

x = torch.arange(start, end, interval).to(device)
x.requires_grad_()
precise = torch.log(x)
precise.sum().backward()

error = (approximate - precise) / precise
print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

error = (approximate - precise)
print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
print("\n")



if args.verbose is True:
    ####################################################################################
    print("# # # # # # # # # # # # # # # #")
    print("# Test RAVENsigmoid")
    print("# # # # # # # # # # # # # # # #")
    start, end, interval = -1., 1., 0.001
    print("input range: ", start, end)
    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    approximate = RAVENsigmoid(cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(x)
    approximate.sum().backward()

    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    precise = torch.sigmoid(x)
    precise.sum().backward()

    error = (approximate - precise) / precise
    print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

    error = (approximate - precise)
    print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
    print("\n")



    ####################################################################################
    print("# # # # # # # # # # # # # # # #")
    print("# Test RAVENtanh")
    print("# # # # # # # # # # # # # # # #")
    start, end, interval = -1., 1., 0.001
    print("input range: ", start, end)
    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    approximate = RAVENtanh(cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(x)
    approximate.sum().backward()

    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    precise = torch.tanh(x)
    precise.sum().backward()

    error = (approximate - precise) / precise
    print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

    error = (approximate - precise)
    print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
    print("\n")



    ####################################################################################
    print("# # # # # # # # # # # # # # # #")
    print("# Test RAVENsoftmax")
    print("# # # # # # # # # # # # # # # #")
    start, end, interval = -1., 1., 0.1
    print("input range: ", start, end)
    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    approximate = RAVENsoftmax(dim=0, cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(x)
    approximate.sum().backward()

    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    precise = torch.nn.Softmax(dim=0)(x)
    precise.sum().backward()

    error = (approximate - precise) / precise
    print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

    error = (approximate - precise)
    print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
    print("\n")



    ####################################################################################
    print("# # # # # # # # # # # # # # # #")
    print("# Test RAVENlogsoftmax")
    print("# # # # # # # # # # # # # # # #")
    start, end, interval = -1., 1., 0.1
    print("input range: ", start, end)
    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    approximate = RAVENlogsoftmax(dim=0, cycle=cycle, intwidth_max=intwidth_max, fracwidth_max=fracwidth_max, bitwidth_reduce=bitwidth_reduce, rounding=rounding).to(device)(x)
    approximate.sum().backward()

    x = torch.arange(start, end, interval).to(device)
    x.requires_grad_()
    precise = torch.nn.LogSoftmax(dim=0)(x)
    precise.sum().backward()

    error = (approximate - precise) / precise
    print("relative error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())

    error = (approximate - precise)
    print("absolute error: %1.3f ~ %1.3f (mean %1.3f)" % (error.min().item(), error.max().item(), error.mean().item()))
    print("\trms error: %1.3f" % (error.mul(error).mean().sqrt()).item())
    print("\n")
