import math
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)

def gen_trace_conv2d_os_read(
        mac_rows = 4, # row size of systolic array
        mac_cols = 4, # column size of systolic array
        batch = 1, # batch size
        ifm_h = 7, # input feature map height
        ifm_w = 7, # input feature map width
        wgt_h = 3, # weight height
        wgt_w = 3, # weight width
        ichl = 3, # input channel count
        ochl = 3, # output channel count, also the filter count
        stride_h = 1, # stride on height dimension, 
        stride_w = 1, # stride on width dimension, 
        ifm_base = 0, # input feature map base addr
        wgt_base = 1000000, # weight base addr
        fm_fmt = "NHWC", # format of feature map layout. options: "NHWC", "NCHW". default is "NHWC".
        sram_trace_conv2d_read_file = "sram_trace_conv2d_read.csv"
    ):
    
    # shared parameters
    # from https://cs231n.github.io/convolutional-networks/#conv
    ofm_h = int((ifm_h - wgt_h) / stride_h + 1)
    ofm_w = int((ifm_w - wgt_w) / stride_w + 1)
    
    conv_size_chl = wgt_h * wgt_w
    conv_size = conv_size_chl * ichl

    ofm_size_chl = ofm_h * ofm_w
    ofm_size = ofm_size_chl * ochl
    
    # this OS dataflow map each ofm channel to a column
    # different ofm pixels of the same channel are folded
    fold_row = int(math.ceil(ofm_size_chl/mac_rows))
    # different ofm channels are folded 
    fold_col = int(math.ceil(ochl/mac_cols))

    

    pass


def gen_trace_conv2d_os_write(
        mac_rows = 4, # row size of systolic array
        mac_cols = 4, # column size of systolic array
        batch = 1, # batch size
        ifm_h = 7, # input feature map height
        ifm_w = 7, # input feature map width
        wgt_h = 3, # weight height
        wgt_w = 3, # weight width
        ichl = 3, # input channel count
        ochl = 3, # output channel count, also the filter count
        stride_h = 1, # stride on height dimension, 
        stride_w = 1, # stride on width dimension, 
        ofm_base = 2000000, # output feature map base addr
        fm_fmt = "NHWC", # format of feature map layout. options: "NHWC", "NCHW". default is "NHWC".
        sram_trace_conv2d_write_file = "sram_trace_conv2d_write.csv"
):