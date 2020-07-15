from gen_trace_conv2d_os import gen_trace_conv2d_os


def sram_trace_conv2d(
        mac_rows = 4, # row size of systolic array
        mac_cols = 4, # column size of systolic array
        batch = 1, # batch size
        ifm_h = 7, # input feature map height
        ifm_w = 7, # input feature map width
        wgt_h = 3, # weight height
        wgt_w = 3, # weight width
        ichn = 2, # input channel count
        ochn = 5, # output channel count, also the filter count
        stride_h = 1, # stride on height dimension
        stride_w = 1, # stride on width dimension
        ifm_base = 0, # input feature map base addr
        wgt_base = 1000000, # weight base addr
        ofm_base = 2000000, # output feature map base addr
        fm_fmt = "NHWC", # format of feature map layout. options: "NHWC", "NCHW". default is "NHWC"
        dataflow = "OS", # data flow in the systolic array. options: "OS", "WS", "IS". default is "OS"
        sram_trace_conv2d_read_file = "sram_trace_conv2d_read.csv", 
        sram_trace_conv2d_write_file = "sram_trace_conv2d_write.csv"
    ):

    """
    This sram_trace_conv2d assumes valid conv, and does not support dilation.
    """

    if dataflow is "OS":
        cycles, utilization = gen_trace_conv2d_os(
                                mac_rows = mac_rows, 
                                mac_cols = mac_cols, 
                                batch = batch, 
                                ifm_h = ifm_h, 
                                ifm_w = ifm_w, 
                                wgt_h = wgt_h, 
                                wgt_w = wgt_w, 
                                ichn = ichn, 
                                ochn = ochn, 
                                stride_h = stride_h, 
                                stride_w = stride_w, 
                                ifm_base = ifm_base, 
                                wgt_base = wgt_base, 
                                ofm_base = ofm_base, 
                                fm_fmt = fm_fmt, 
                                sram_trace_conv2d_read_file = sram_trace_conv2d_read_file,
                                sram_trace_conv2d_write_file = sram_trace_conv2d_write_file
                                )

    elif dataflow is "IS":
        raise ValueError("Input dataflow -> "+dataflow+" <- is not implemented yet.")
    elif dataflow is "WS":
        raise ValueError("Input dataflow -> "+dataflow+" <- is not implemented yet.")
    else:
        raise ValueError("Input dataflow -> "+dataflow+" <- is not supported.")

    return cycles, utilization


if __name__ == "__main__":
   sram_trace_conv2d()

