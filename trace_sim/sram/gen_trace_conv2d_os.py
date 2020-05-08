import math
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def gen_trace_conv2d_os(
        mac_rows = 4, # row size of systolic array
        mac_cols = 4, # column size of systolic array
        batch = 2, # batch size
        ifm_h = 7, # input feature map height
        ifm_w = 7, # input feature map width
        wgt_h = 3, # weight height
        wgt_w = 3, # weight width
        ichn = 3, # input channel count
        ochn = 3, # output channel count, also the filter count
        stride_h = 1, # stride on height dimension
        stride_w = 1, # stride on width dimension
        ifm_base = 0, # input feature map base addr
        wgt_base = 1000000, # weight base addr
        ofm_base = 2000000, # output feature map base addr
        fm_fmt = "NHWC", # format of feature map layout, options: "NHWC", "NCHW", default is "NHWC"
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

    ofm_size_chn = ofm_h * ofm_w
    ofm_size = batch * ofm_size_chn * ochn
    
    conv_size_chn = wgt_h * wgt_w
    conv_size_px = conv_size_chn * ichn

    total_read_cnt = ofm_size * conv_size_px
    total_write_cnt = ofm_size * 1

    # flag if all ofm pixels are done
    all_px_read_done = False
    ifm_read_cnt = 0
    all_px_write_done = False
    ofm_write_cnt = 0

    # parameter for folding boundry check
    row_fold_chn = int(ofm_size_chn % mac_rows)
    col_fold_chn = int((batch * ochn) % mac_cols)
    
    # pbar = tqdm(total=batch * ochn)

    # the ofm batch index that each column of pes is assigned for computing
    remained_nc = []
    for n in range(batch):
        for c in range(ochn):
            remained_nc.append(n * ochn + c)

    act_cols = np.array([[x < min(len(remained_nc), mac_cols) for x in range(mac_cols)]], dtype=int)
    remained_nc = np.array(remained_nc, dtype=int)
    remained_nc = remained_nc[int(np.sum(act_cols)):]

    act_rows = np.array([[x < min(ofm_size_chn, mac_rows)] for x in range(mac_rows)], dtype=int)

    act_idx = np.zeros((mac_rows, mac_cols), dtype=bool)
    for r in range(np.sum(act_rows)):
        for c in range(np.sum(act_cols)):
            act_idx[r, c] = True

    init_col_idx = np.zeros(mac_cols, dtype=int) \
                 + np.array([c for c in range(np.sum(act_cols))] + [0 for c in range(mac_cols - np.sum(act_cols))], dtype=int)
    i_col_ofm_n = np.floor(init_col_idx / ochn).astype("int")

    # the ofm channel index that each column of pes is assigned for computing
    i_col_ofm_c = init_col_idx % ochn
    # the ofm h index that each pe is assigned for computing
    i_row_ofm_h = np.zeros((mac_rows, mac_cols), dtype=int)
    # the ofm w index that each pe is assigned for computing
    i_row_ofm_w = np.zeros((mac_rows, mac_cols), dtype=int) \
                + np.array([[r] for r in range(np.sum(act_rows))] + [[0] for c in range(mac_rows - np.sum(act_rows))], dtype=int)

    # the weight input channel index that each pe is assigned for computing
    i_row_wgt_ci = np.zeros((mac_rows, mac_cols), dtype=int)
    for r in range(mac_rows):
        for c in range(mac_cols):
            i_row_wgt_ci[r, c] = 0 - r - c

    # the weight h index that each pe is assigned for computing
    # as width dimension is updated before height dimension, height is initialized to 0
    i_row_wgt_h = np.zeros((mac_rows, mac_cols), dtype=int)
    # the weight w index that each pe is assigned for computing
    # width dimension is used to determine whether a pe is ocuppied by a non-negative element
    i_row_wgt_w = np.zeros((mac_rows, mac_cols), dtype=int)

    # the ofm batch index that each pe is assigned for output
    o_col_ofm_n = np.zeros(mac_cols, dtype=int)
    # the ofm channel index that each column of pes is assigned for output
    o_col_ofm_c = np.zeros(mac_cols, dtype=int)
    # the ofm h index that each pe is assigned for output
    o_row_ofm_h = np.zeros((mac_rows, mac_cols), dtype=int)
    # the ofm w index that each pe is assigned for output
    o_row_ofm_w = np.zeros((mac_rows, mac_cols), dtype=int)
    # indicate which pixel to output
    out_vld = np.zeros((mac_rows, mac_cols), dtype=bool)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # implementing different mapping strategies
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def mapping_0(
            remained_nc, act_idx, \
            i_col_ofm_n, i_col_ofm_c, i_row_ofm_h, i_row_ofm_w, \
            i_row_wgt_ci, i_row_wgt_h, i_row_wgt_w, \
            o_col_ofm_n, o_col_ofm_c, o_row_ofm_h, o_row_ofm_w
        ):
        # this OS dataflow map each ofm channel to a column
        # different ofm pixels of the same channel are folded
        # different ofm channels are folded

        # each column is assigned to a different output channel, and there are two candicate strategies:
        # 1. all output channels with the same index at different batches are mapped to the same column.
        # 2. all output channels of the same batches are mapped to the different columns. Less sram read cost for weight.
        # this mapping function implements strategy 2

        # get the ofm index before update calculation
        o_col_ofm_n = deepcopy(i_col_ofm_n)
        o_col_ofm_c = deepcopy(i_col_ofm_c)
        o_row_ofm_h = deepcopy(i_row_ofm_h)
        o_row_ofm_w = deepcopy(i_row_ofm_w)

        # processing one pixel
        # assume different channels are read most frequently, as it's NHWC for feature map
        i_row_wgt_ci[act_idx] += 1
        # i_row_wgt_ci_carry indicates that all ifm channels of an ofm pixel are done
        i_row_wgt_ci_carry = (np.floor(i_row_wgt_ci / ichn) == 1) * act_idx
        i_row_wgt_ci[i_row_wgt_ci_carry] = 0
        i_row_wgt_ci[np.invert(act_idx)] = -1

        i_row_wgt_w[i_row_wgt_ci_carry] += 1
        # i_row_wgt_w_carry indicates that one ifm row of an ofm pixel is done
        i_row_wgt_w_carry = (np.floor(i_row_wgt_w / wgt_w) == 1) * act_idx
        i_row_wgt_w[i_row_wgt_w_carry] = 0

        i_row_wgt_h[i_row_wgt_w_carry] += 1
        # i_row_wgt_h_carry indicates that one ofm pixel is done
        i_row_wgt_h_carry = (np.floor(i_row_wgt_h / wgt_h) == 1) * act_idx
        i_row_wgt_h[i_row_wgt_h_carry] = 0

        out_vld = i_row_wgt_h_carry

        # processing next pixel in the same channel
        i_row_ofm_w[i_row_wgt_h_carry] += mac_rows
        # i_row_ofm_w_carry indicates that one ofm row is done
        i_row_ofm_w_carry = (np.floor(i_row_ofm_w / ofm_w) == 1) * act_idx
        i_row_ofm_w[i_row_ofm_w_carry] = i_row_ofm_w[i_row_ofm_w_carry] % ofm_w

        i_row_ofm_h[i_row_ofm_w_carry] += 1
        # i_row_ofm_h_carry indicates that one ofm channel is done
        i_row_ofm_h_carry = (np.floor(i_row_ofm_h / ofm_h) == 1) * act_idx
        i_row_ofm_h[i_row_ofm_h_carry] = 0


        next_px = i_row_ofm_h * ofm_w \
                   + i_row_ofm_w
        diff_px = ofm_size_chn - next_px - 1 == 0
        act_idx[ofm_size_chn - next_px - 1 < 0] = 0
        for r in range(mac_rows-2, -1, -1):
            for c in range(mac_rows-1, -1, -1):
                if act_idx[r, c] == 0 or diff_px[r, c] == 1:
                    act_idx[r+1, c] = False
                    
        # processing next channel
        # print("i_row_ofm_h_carry\n", i_row_ofm_h_carry)
        # print("i_row_ofm_h_carry[0, :]\n", i_row_ofm_h_carry[0, :])
        top_sum = np.sum(i_row_ofm_h_carry[0, :].astype("int"))
        assert top_sum <= 1, "sum of i_row_ofm_h_carry[0] should be no greater than 1."
        # i_col_ofm_n[i_row_ofm_h_carry[0, :]] = np.floor(remained_nc[0] / ochn)
        # i_col_ofm_c[i_row_ofm_h_carry[0, :]] = remained_nc[0] % ochn
        if len(remained_nc) > 0:
            if top_sum == 1:
                # if more remained channels, assign channel and update act_cols
                i_col_ofm_n[i_row_ofm_h_carry[0, :]] = int(np.floor(remained_nc[0] / ochn))
                i_col_ofm_c[i_row_ofm_h_carry[0, :]] = int(remained_nc[0] % ochn)

                act_rows_temp = min(ofm_size_chn, mac_rows)
                for c in range(len(i_row_ofm_h_carry[0, :])):
                    if i_row_ofm_h_carry[0, c] == True:
                        new_cols = c
                act_idx[0:act_rows_temp, new_cols] = True
                act_idx[act_rows_temp:, new_cols] = False
                i_row_ofm_h[:, new_cols] = 0
                i_row_wgt_h[:, new_cols] = 0
                i_row_wgt_w[:, new_cols] = 0

                i_row_ofm_w[0:act_rows_temp, new_cols] = np.array([r for r in range(act_rows_temp)])
                i_row_wgt_ci[0:act_rows_temp, new_cols] = np.array([-r for r in range(act_rows_temp)])
                
                remained_nc = remained_nc[1:]
        else:
            # if no more remained channels, deactivate the finished column
            act_idx[i_row_ofm_h_carry] = 0

        return remained_nc, act_idx, \
                i_col_ofm_n, i_col_ofm_c, i_row_ofm_h, i_row_ofm_w, \
                i_row_wgt_ci, i_row_wgt_h, i_row_wgt_w, \
                o_col_ofm_n, o_col_ofm_c, o_row_ofm_h, o_row_ofm_w, out_vld

    while not all_px_write_done:
        print("cycles===============>", cycles)
        if fm_fmt is "NHWC":
            # all indexes start from 0, negative means not mapped.
            # print("remained_nc\n", remained_nc)
            # print("act_idx\n", act_idx)
            # print("i_col_ofm_n\n", i_col_ofm_n)
            # print("i_col_ofm_c\n", i_col_ofm_c)
            # print("i_row_ofm_h\n", i_row_ofm_h)
            # print("i_row_ofm_w\n", i_row_ofm_w)
            # print("i_row_wgt_h\n", i_row_wgt_h)
            # print("i_row_wgt_w\n", i_row_wgt_w)
            # print("i_row_wgt_ci\n", i_row_wgt_ci)
            # print("o_col_ofm_n\n", o_col_ofm_n)
            # print("o_col_ofm_c\n", o_col_ofm_c)
            # print("o_row_ofm_h\n", o_row_ofm_h)
            # print("o_row_ofm_w\n", o_row_ofm_w)

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # weight trace from the top of the wgt_addr
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            """
            for weight, even if the feature map memory organization is fixed, weight memory organization can vary:
                offset_ohwi(c_o, h, w, c_i) = c_o * HWC_i + h * WC_i + w * C_i + c_i
            """
            wgt_addr = i_col_ofm_c * wgt_h * wgt_w * ichn \
                     + i_row_wgt_h * wgt_w * ichn \
                     + i_row_wgt_w * ichn \
                     + i_row_wgt_ci
            
            # print("wgt_addr\n", wgt_addr)
            wgt_read = ""
            for c in range(mac_cols):
                if i_row_wgt_ci[0, c] < 0 and act_idx[0, c] == 1:
                    wgt_read += ", "
                else:
                    wgt_read += str(int(wgt_addr[0, c] + wgt_base)) + ", "

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # ifm trace from the left of the ifm_addr
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            """
            https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html
            for fm:
                offset_nhwc(n, h, w, c) = n * HWC + h * WC + w * C + c
            """
            ifm_addr = i_col_ofm_n * ifm_h * ifm_w * ichn \
                     + (i_row_ofm_h * stride_h + i_row_wgt_h) * ifm_w * ichn \
                     + (i_row_ofm_w * stride_w + i_row_wgt_w) * ichn \
                     + i_row_wgt_ci

            # print("ifm_addr\n", ifm_addr)
            ifm_read = ""
            for r in range(mac_rows):
                if i_row_wgt_ci[r, 0] < 0:
                    ifm_read += ", "
                else:
                    ifm_read += str(int(ifm_addr[r, 0] + ifm_base)) + ", "
                    ifm_read_cnt += 1

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # ofm trace by checking the valid bottom of the mapping matrix
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            """
            https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html
            for fm:
                offset_nhwc(n, h, w, c) = n * HWC + h * WC + w * C + c
            """
            ofm_addr = o_col_ofm_n * ofm_h * ofm_w * ochn \
                     + o_row_ofm_h * ofm_w * ochn \
                     + o_row_ofm_w * ochn \
                     + o_col_ofm_c
            
            print(out_vld)
            ofm_write = ""
            col_vld = np.sum(out_vld, axis=0)
            ofm_vld = np.sum(col_vld)

            if ofm_vld != 0:
                for c in range(mac_cols):
                    assert col_vld[c] <= 1, "One column should not output two ofm at the same cycle."
                    if col_vld[c] == 1:
                        for r in range(mac_rows):
                            if out_vld[r, c] == 1:
                                out_row = r
                        ofm_write += str(int(ofm_addr[out_row, c] + ofm_base)) + ", "
                        ofm_write_cnt += 1
                    else:
                        ofm_write += ", "

            # generate trace for read and write at each cycle
            print("ifm_read_cnt", ifm_read_cnt)
            print("total_read_cnt", total_read_cnt)
            if ifm_read_cnt >= total_read_cnt:
                all_px_read_done = True

            if not all_px_read_done:
                read_trace = str(cycles) + ", " + ifm_read + wgt_read + "\n"
                read_outfile.write(read_trace)
                read_cycles += 1

            print("ofm_write_cnt", ofm_write_cnt)
            print("total_write_cnt", total_write_cnt)
            if ofm_write_cnt >= total_write_cnt:
                all_px_write_done = True
                
            if not all_px_write_done and ofm_vld != 0:
                write_trace = str(cycles) + ", " + ofm_write + "\n"
                write_outfile.write(write_trace)
                write_cycles += 1

            # calculate the percentage of PE utilization at each cycle
            val_pe_count = np.sum((i_row_ofm_h > 0).astype('int'))
            utilization += val_pe_count/mac_rows/mac_cols

            cycles += 1

            # if not all ofm pixels are done, according to the border information of systolic array, generate the trace
            remained_nc, act_idx, \
            i_col_ofm_n, i_col_ofm_c, i_row_ofm_h, i_row_ofm_w, \
            i_row_wgt_ci, i_row_wgt_h, i_row_wgt_w, \
            o_col_ofm_n, o_col_ofm_c, o_row_ofm_h, o_row_ofm_w, out_vld = mapping_0(
                                                                            remained_nc, act_idx, \
                                                                            i_col_ofm_n, i_col_ofm_c, i_row_ofm_h, i_row_ofm_w, \
                                                                            i_row_wgt_ci, i_row_wgt_h, i_row_wgt_w, \
                                                                            o_col_ofm_n, o_col_ofm_c, o_row_ofm_h, o_row_ofm_w
                                                                            )

        elif fm_fmt is "NCHW":
            """
            offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
            """
            # TBD
            raise ValueError("Input feature map format -> "+fm_fmt+" <- is not implemented yet.")

    # utilization /= read_cycles + 1
    utilization /= cycles + 1
    
    # pbar.close()
    read_outfile.close()
    write_outfile.close()

    return cycles, utilization
