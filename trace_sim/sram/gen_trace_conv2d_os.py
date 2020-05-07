import math
import numpy as np
from tqdm import tqdm


def gen_trace_conv2d_os(
        mac_rows = 4, # row size of systolic array
        mac_cols = 4, # column size of systolic array
        batch = 2, # batch size
        ifm_h = 7, # input feature map height
        ifm_w = 7, # input feature map width
        wgt_h = 3, # weight height
        wgt_w = 3, # weight width
        ichl = 3, # input channel count
        ochl = 5, # output channel count, also the filter count
        stride_h = 1, # stride on height dimension
        stride_w = 1, # stride on width dimension
        ifm_base = 0, # input feature map base addr
        wgt_base = 1000000, # weight base addr
        ofm_base = 2000000, # output feature map base addr
        fm_fmt = "NHWC", # format of feature map layout, options: "NHWC", "NCHW", default is "NHWC"
        stall = 0, # the stall cycles between switching mapping, default 0
        sram_trace_conv2d_read_file = "sram_trace_conv2d_os_read.csv",
        sram_trace_conv2d_write_file = "sram_trace_conv2d_os_write.csv"
    ):

    # output parameters
    cycles = 0
    read_cycles = 0
    write_cycles = 0
    # PE array utilization
    utilization = 0
    read_outfile = open(sram_trace_conv2d_read_file, "w")
    write_outfile = open(sram_trace_conv2d_write_file, "w")

    # shared parameters: 
    # 1. output feature map height/width
    # 2. convolution window size
    # from https://cs231n.github.io/convolutional-networks/#conv
    ofm_h = int((ifm_h - wgt_h) / stride_h + 1)
    ofm_w = int((ifm_w - wgt_w) / stride_w + 1)
    
    ofm_size_chl = ofm_h * ofm_w
    ofm_size = ofm_size_chl * ochl
    
    conv_size_chl = wgt_h * wgt_w
    conv_size_px = conv_size_chl * ichl

    total_read_cnt = ofm_size * conv_size_px
    total_write_cnt = ofm_size * 1

    # flag if all ofm pixels are done
    all_px_read_done = False
    ifm_read_cnt = 0
    all_px_write_done = False
    ofm_write_cnt = 0

    # parameter for trace generation
    # the ofm batch index that each column of pes is assigned for computing
    total_map_nc = []
    for n in range(batch):
        for c in range(ichl):
            total_map_nc.append(n * ichl + c + 1)
    total_map_nc = np.array(total_map_nc, dtype=float)
    act_cols = min(ochl, mac_cols)
    init_col_idx = np.zeros(mac_cols) \
                 + np.array([c for c in range(act_cols)] + [0 for c in range(mac_cols - act_cols)], dtype=float)
    i_col_ofm_n = np.floor(init_col_idx/ichl)
    print(i_col_ofm_n)
    # the ofm channel index that each column of pes is assigned for computing
    i_col_ofm_c = init_col_idx%ichl
    print(i_col_ofm_c)
    # the ofm h index that each pe is assigned for computing
    i_row_ofm_h = np.zeros((mac_rows, mac_cols))
    # the ofm w index that each pe is assigned for computing
    i_row_ofm_w = np.zeros((mac_rows, mac_cols))

    # the weight input channel index that each pe is assigned for computing
    i_row_wgt_ci = np.zeros((mac_rows, mac_cols))
    # the weight h index that each pe is assigned for computing
    i_row_wgt_h = np.zeros((mac_rows, mac_cols))
    # the weight w index that each pe is assigned for computing
    i_row_wgt_w = np.zeros((mac_rows, mac_cols))

    # the ofm batch index that each pe is assigned for output
    o_col_ofm_n = np.zeros(mac_cols)
    # the ofm channel index that each column of pes is assigned for output
    o_col_ofm_c = np.zeros(mac_cols)
    # the ofm h index that each pe is assigned for output
    o_row_ofm_h = np.zeros((mac_rows, mac_cols))
    # the ofm w index that each pe is assigned for output
    o_row_ofm_w = np.zeros((mac_rows, mac_cols))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # implementing different mapping strategies
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def mapping_NHWC(
            i_col_ofm_n, 
            i_col_ofm_c, 
            i_row_ofm_h, 
            i_row_ofm_w, 
            i_row_wgt_ci, 
            i_row_wgt_h, 
            i_row_wgt_w, 
            o_col_ofm_n, 
            o_col_ofm_c, 
            o_row_ofm_h, 
            o_row_ofm_w, 

        ):
        # this OS dataflow map each ofm channel to a column
        # different ofm pixels of the same channel are folded
        fold_row = int(math.ceil(ofm_size_chl/mac_rows))
        # different ofm channels are folded
        fold_col = int(math.ceil(ochl/mac_cols))

        # each column is assigned to a different output channel, and there are two candicate strategies:
        # 1. all output channels with the same index at different batches are mapped to the same column.
        # 2. all output channels of the same batches are mapped to the different columns. Less sram read cost for weight.

        # this mapping implements strategy 2


        # initial mapping of the systolic array col to ofm channel index at the current cycle
        # if the output channel count is less than the column count
        act_cols = min(ochl, mac_cols)
        col_batch_map = np.zeros(mac_cols) \
                      + np.array([c for c in range(1, act_cols+1)] + [0 for c in range(mac_cols - act_cols)], dtype=float)
        
        # mapping of the row pe to ofm pixel index at the current cycle
        act_rows = min(ofm_size_chl, mac_rows)
        row_map = np.zeros((mac_rows, mac_cols)) \
                + np.transpose(np.tile(np.array([r for r in range(1, act_rows+1)] + [0 for r in range(mac_rows - act_rows)], dtype=float), (mac_rows, 1))) \
                * np.array([1 for c in range(1, act_cols+1)] + [0 for c in range(mac_cols - act_cols)], dtype=float)
        
        
        print("col_mapping:\n", col_mapping)
        print("row_mapping:\n", row_mapping)

        return col_map_bth, col_map_chl, row_map_h, row_map_w, row_cycle_h, row_cycle_w

    while not all_px_write_done:
        if fm_fmt is "NHWC":
            # all indexes start from 0, negative means not mapped.

            # if not all ofm pixels are done, according to the border information of systolic array, generate the trace
            i_col_ofm_n, i_col_ofm_c, i_row_ofm_h, i_row_ofm_w, \
            i_row_wgt_ci, i_row_wgt_h, i_row_wgt_w, \
            o_col_ofm_n, o_col_ofm_c, o_row_ofm_h, o_row_ofm_w = mapping_NHWC(
                                                                    i_col_ofm_n, 
                                                                    i_col_ofm_c, 
                                                                    i_row_ofm_h, 
                                                                    i_row_ofm_w, 
                                                                    i_row_wgt_ci, 
                                                                    i_row_wgt_h, 
                                                                    i_row_wgt_w, 
                                                                    o_col_ofm_n, 
                                                                    o_col_ofm_c, 
                                                                    o_row_ofm_h, 
                                                                    o_row_ofm_w
                                                                    )

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # weight trace from the top of the wgt_addr
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            """
            for weight, even if the feature map memory organization is fixed, weight memory organization can vary:
                offset_ohwi(c_o, h, w, c_i) = c_o * HWC_i + h * WC_i + w * C_i + c_i
            """
            wgt_addr = i_col_ofm_c * wgt_h * wgt_w * ichl \
                     + i_row_wgt_h * wgt_w * ichl \
                     + i_row_wgt_w * ichl \
                     + i_row_wgt_ci

            wgt_read = ""
            for c in range(mac_cols):
                if i_row_wgt_w[0, c] < 0:
                    wgt_read += ", "
                else:
                    wgt_read += str(wgt_addr[r, 0]) + ", "

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # ifm trace from the left of the ifm_addr
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            """
            https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html
            for fm:
                offset_nhwc(n, h, w, c) = n * HWC + h * WC + w * C + c
            """
            ifm_addr = i_col_ofm_n * ifm_h * ifm_w * ichl \
                     + (i_row_ofm_h * stride_h + i_row_wgt_h) * ifm_w * ichl \
                     + (i_row_ofm_w * stride_w + i_row_wgt_w) * ichl \
                     + i_col_ofm_c

            ifm_read = ""
            for r in range(mac_rows):
                if i_row_ofm_w[r, 0] < 0:
                    ifm_read += ", "
                else:
                    ifm_read += str(ifm_addr[r, 0]) + ", "
                    ifm_read_cnt += 1

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # ofm trace by checking the valid bottom of the mapping matrix
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            """
            https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html
            for fm:
                offset_nhwc(n, h, w, c) = n * HWC + h * WC + w * C + c
            """
            ofm_addr = o_col_ofm_n * ofm_h * ofm_w * ochl \
                     + o_row_ofm_h * ofm_w * ochl \
                     + o_row_ofm_w * ochl \
                     + o_col_ofm_c
            
            ofm_write = ""
            for c in range(mac_cols):
                if o_row_ofm_w[0, c] < 0:
                    ofm_write += ", "
                else:
                    ofm_write += str(ofm_addr[0, c]) + ", "
                    ofm_write_cnt += 1

            # generate trace for read and write at each cycle
            if ifm_read_cnt >= total_read_cnt:
                all_px_read_done = True

            if not all_px_read_done:
                read_trace = str(cycles) + ", " + ifm_read + wgt_read
                read_outfile.write(read_trace)
                read_cycles += 1

            if ofm_write_cnt >= total_write_cnt:
                all_px_read_done = True
                
            if not all_px_read_done:
                write_trace = str(cycles) + ", " + ofm_write
                write_outfile.write(write_trace)
                write_cycles += 1

            # calculate the percentage of PE utilization at each cycle
            val_pe_count = np.sum((i_row_ofm_h > 0).astype('float'))
            utilization += val_pe_count/mac_rows/mac_cols

            cycles += 1

        elif fm_fmt is "NCHW":
            """
            offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
            """
            # TBD
            raise ValueError("Input feature map format -> "+fm_fmt+" <- is not implemented yet.")

    # utilization /= read_cycles + 1
    utilization /= cycles + 1

    return cycles, utilization
