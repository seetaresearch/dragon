// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_

#include "core/operator.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
class ConvOpBase : public Operator<Context> {
 public:
    ConvOpBase(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          data_format(OperatorBase::GetSingleArg<string>("data_format", "NCHW")),
          padding(OperatorBase::GetSingleArg<string>("padding", "VALID")),
          num_output(OperatorBase::GetSingleArg<int>("num_output", 1)),
          group(OperatorBase::GetSingleArg<int>("group", 1)),
          static_dsize(OperatorBase::GetRepeatedArg<int>("static_dsize")),
          dynamic_dsize(OperatorBase::GetRepeatedArg<string>("dynamic_dsize")) {
        if (data_format == "NCHW") spatial_axis = 2;
        else if (data_format == "NHWC") spatial_axis = 1;
        else LOG(FATAL) << "Unknown data format: " << data_format;
        num_spatial_axes = -1;  // unknown
    }

 protected:
    vector<TIndex> kernel_size, stride, pad, dilation;
    string data_format, padding;
    vector<TIndex> input_shape, output_shape, bottom_shape, top_shape, col_shape;
    vector<TIndex> weight_shape, bias_shape;
    Tensor* col_buffer, *bias_multiplier;
    TIndex num_output, group;
    TIndex spatial_axis, num_spatial_axes;
    TIndex channels, out_spatial_dim;
    TIndex conv_in_channels, conv_out_channels;
    TIndex conv_out_spatial_dim, kernel_dim;
    TIndex col_offset, output_offset, weight_offset, x_offset, y_offset;
    vector<int> static_dsize;
    vector<string> dynamic_dsize;
    bool is_1x1;

    void Setup();
    void Reshape();
    void GradientReshape();
    virtual void ComputeOutputShape();
    virtual bool ReverseDimensions() = 0;

    template <typename T> void Wx(const T* x, const T* weights, T* y, bool skip_im2col = false);
    template <typename T> void Pb(const T* bias, T* y);
    template <typename T> void Dx(const T* dy, const T* weights, T* dx);
    template <typename T> void Dw(const T* dy, const T* x, T *dw);
    template <typename T> void Db(const T* dy, T* db);

 private:
    template <typename T> void Im2Col(const T* im, T* col) {
        if (input(0).ndim() == 4) {
             kernel::Im2Col2d<T, Context>(conv_in_channels,
                            input_shape[0], input_shape[1],
                          output_shape[0], output_shape[1],
                            kernel_size[0], kernel_size[1],
                                      stride[0], stride[1],
                                            pad[0], pad[1],
                                  dilation[0], dilation[1],
                                               data_format,
                                                        im,
                                                      col);
        } else LOG(FATAL) << "ConvNd has not been implemented yet";
    }
    template <typename T> void Col2Im(const T* col, T* im) {
        if (input(0).ndim() == 4) {
             kernel::Col2Im2d<T, Context>(conv_in_channels,
                            input_shape[0], input_shape[1],
                          output_shape[0], output_shape[1],
                            kernel_size[0], kernel_size[1],
                                      stride[0], stride[1],
                                            pad[0], pad[1],
                                  dilation[0], dilation[1],
                                               data_format,
                                                       col,
                                                       im);
        } else LOG(FATAL) << "ConvNd has not been implemented yet";
    }
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_