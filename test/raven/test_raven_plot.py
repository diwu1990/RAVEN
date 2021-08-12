import matplotlib.pyplot as plt

# (int, frac) : [1-7-8, 1-6-9, 1-5-10, 1-4-11]
# cycle=4, bitwidth-reduce=False
config1 = "4-4"
div_acc = [0.022, 0.022, 0.021, 0.021]
exp_acc = [0.079, 0.079, 0.079, 0.079]
log_acc = [0.006, 0.005, 0.005, 0.005]
dnn_acc = [98.686224, 98.871771, 98.758370, 93.980589]

# cycle=4, bitwidth-reduce=True
config2 = "4-4-CLA"
div_acc = [0.023, 0.022, 0.022, 0.022]
exp_acc = [0.080, 0.080, 0.080, 0.080]
log_acc = [0.008, 0.006, 0.005, 0.005]
dnn_acc = [98.702766, 98.871771, 98.745615, 94.051937]

# cycle=8, bitwidth-reduce=False
config3 = "8-8"
div_acc = [0.003, 0.001, 0.001, 0.000]
exp_acc = [0.003, 0.001, 0.001, 0.000]
log_acc = [0.002, 0.001, 0.000, 0.000]
dnn_acc = [98.680245, 98.853037, 98.777901, 93.784479]

# cycle=8, bitwidth-reduce=True
config4 = "8-8-CLA"
div_acc = [0.158, 0.078, 0.039, 0.019]
exp_acc = [0.129, 0.064, 0.032, 0.016]
log_acc = [0.103, 0.051, 0.026, 0.013]
dnn_acc = [86.265944, 97.750319, 98.355788, 93.550104]

fp_dnn_acc = [98.780891]