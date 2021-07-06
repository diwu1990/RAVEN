# %%
import torch
from RAVEN.pe.uno.ops import *
import time

# %%
"""
# test exp operator
"""

# %%
a = torch.arange(0, 1, 0.001)
a = a.cuda()

start_time = time.time()
approximate = UNOexp.apply(a, 5, 7, 8, True)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
precise = torch.exp(a)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
"""
# reference result using UNOtaylor kernel
"""

# %%
a = torch.arange(0, 1, 0.001)
a = a.cuda()

precise = torch.exp(a)

point = 0.
scale = torch.exp(torch.tensor([point])).cuda()
var = a - point

coeff = [1/1, 1/1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320, 1/362880]

coeff = coeff[0:5]
print(coeff)

approximate = UNOtaylor(scale, 
                         coeff, 
                         var, 
                         fxp=True, 
                         intwidth=7, 
                         fracwidth=8, 
                         rounding_coeff="round", 
                         rounding_var="round", 
                         keepwidth=False)

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())

# %%
"""
# test exp op grad
"""

# %%
a = torch.tensor([0.5, 0.7]).cuda()
a.requires_grad_()

approximate = UNOexp.apply(a, 5, 7, 8, True, "round", True, True, True)
print(approximate)
approximate.sum().backward()
print(a.grad)

a = torch.tensor([0.5, 0.7])
a.requires_grad_()

precise = torch.exp(a)
precise.sum().backward()
a.grad
print(precise)
print(a.grad)


# %%
"""
# test left most one
"""

# %%
a = torch.tensor([127.5, 0.99])
b = torch.log2(a).floor()
print(a >> (b+1))
print(a)

# %%
"""
# test div grad
"""

# %%
a = torch.tensor([0.5]).cuda()
b = torch.tensor([2.]).cuda()
a.requires_grad_()
b.requires_grad_()

c = torch.div(a, b)
print(c)
c.backward()
print(a.grad)
print(b.grad)

# %%
"""
# test UNOdiv
"""

# %%
num = 100000
y = torch.rand(num).mul(32).cuda()
x = torch.rand(num).mul(64).cuda()

y = torch.tensor([1.]).cuda()
x = torch.arange(0.5, 1., 0.001).cuda()

y = torch.tensor([32.12341, 6.123]).cuda()
x = torch.tensor([0.5123141, 0.1231]).cuda()

y.requires_grad_()
x.requires_grad_()

# print(y, x)

start_time = time.time()
approximate = UNOdiv.apply(y, x, 7, 7, 8, True)
approximate.sum().backward()
print(y.grad)
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

y = torch.tensor([32.12341, 6.123]).cuda()
x = torch.tensor([0.5123141, 0.1231]).cuda()

y.requires_grad_()
x.requires_grad_()

start_time = time.time()
precise = torch.div(y, x)
precise.sum().backward()
print(y.grad)
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
"""
# test UNOlog
"""

# %%
num = 100000
x = torch.rand(num).mul(64).cuda()

x = torch.arange(0.5, 1., 0.001)
# x = x.cuda()

x = torch.tensor([0.5123141, 43]).cuda()

x.requires_grad_()

start_time = time.time()
approximate = UNOlog.apply(x, 8, 7, 8, False)
approximate.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

x = torch.tensor([0.5123141, 43]).cuda()

x.requires_grad_()

start_time = time.time()
precise = torch.log(x)
precise.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
"""
# test UNOsigmoid
"""

# %%
num = 100000
x = torch.rand(num).mul(64).cuda()

x = torch.arange(0.5, 1., 0.001)
# x = x.cuda()

x = torch.tensor([0.5123141, 43.]).cuda()

x.requires_grad_()

start_time = time.time()
approximate = UNOsigmoid(x, 8, 7, 8, True)
approximate.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

x = torch.tensor([0.5123141, 43.]).cuda()

x.requires_grad_()

start_time = time.time()
precise = torch.sigmoid(x)
precise.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
"""
# test UNOtanh
"""

# %%
num = 100000
x = torch.rand(num).mul(64).cuda()

x = torch.arange(0.5, 1., 0.001)
# x = x.cuda()

x = torch.tensor([0.5123141, 2.1]).cuda()

x.requires_grad_()

start_time = time.time()
approximate = UNOtanh(x, 8, 7, 8, True)
approximate.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

x = torch.tensor([0.5123141, 2.1]).cuda()

x.requires_grad_()

start_time = time.time()
precise = torch.tanh(x)
precise.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
"""
# test UNOsoftmax
"""

# %%
x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).cuda()

x.requires_grad_()

dim = 1

start_time = time.time()
approximate = UNOsoftmax(x, dim, 10, 7, 8, False)
approximate.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).cuda()

x.requires_grad_()

start_time = time.time()
softmax = torch.nn.Softmax(dim=dim)
precise = softmax(x)
precise.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
"""
# test UNOlogsoftmax
"""

# %%
x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).cuda()

x.requires_grad_()

dim = 1

start_time = time.time()
approximate = UNOlogsoftmax(x, dim, 8, 7, 8, True)
approximate.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

x = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]).cuda()

x.requires_grad_()

start_time = time.time()
softmax = torch.nn.LogSoftmax(dim=dim)
precise = softmax(x)
precise.sum().backward()
print(x.grad)
print("--- %s seconds ---" % (time.time() - start_time))

error = (approximate - precise) / precise
print("min error rate:", error.min())
print("max error rate:", error.max())
print("avg error rate:", error.mean())
print("rms error rate:", error.mul(error).mean().sqrt())


# %%
