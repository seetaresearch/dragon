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
          num_output(OperatorBase::GetSingleArg<int>("num_output", 1)), 
          group(OperatorBase::GetSingleArg<int>("group", 1)) {

        channel_axis = 1, num_spatial_axes = 2;    // Conv2D support only Now
        vector<TIndex> spatial_shape(1, num_spatial_axes);

        vector<int> ks = OperatorBase::GetRepeatedArg<int>("kernel_size");
        for (int i = 0; i < num_spatial_axes; i++) 
            kernel_size.push_back(i < ks.size() ? ks[i]: ks[0]);

        vector<int> s = OperatorBase::GetRepeatedArg<int>("stride");
        for (int i = 0; i < num_spatial_axes; i++)
            stride.push_back(i < s.size() ? s[i] : s[0]);

        vector<int> p = OperatorBase::GetRepeatedArg<int>("pad");
        for (int i = 0; i < num_spatial_axes; i++)
            pad.push_back(i < p.size() ? p[i] : p[0]);

        vector<int> d = OperatorBase::GetRepeatedArg<int>("dilation");
        for (int i = 0; i < num_spatial_axes; i++)
            dilation.push_back(i < d.size() ? d[i] : d[0]);

        is_1x1 = true;
        for (int i = 0; i < num_spatial_axes; i++) {
            is_1x1 &= (kernel_size[i] == 1 && 
                       stride[i] == 1 && 
                       pad[i] == 0);
            if (!is_1x1) break;
        }
    }

 protected:
    vector<TIndex> kernel_size, stride, pad, dilation;
    vector<TIndex> input_shape, output_shape, bottom_shape, col_buffer_shape;
    vector<TIndex> weight_shape, bias_shape;
    Tensor* col_buffer, *bias_multiplier;
    TIndex num_output, group;
    TIndex channel_axis, num_spatial_axes;
    TIndex channels, out_spatial_dim;
    TIndex conv_in_channels, conv_out_channels;
    TIndex conv_out_spatial_dim, kernel_dim;
    TIndex col_offset, output_offset, weight_offset, x_offset, y_offset;
    bool is_1x1;

    void Reshape();
    void GradientReshape();
    virtual void ComputeOutputShape() = 0;
    virtual bool ReverseDimensions() = 0;

    template <typename T> void Wx(const T* x, const T* weights, T* y, bool skip_im2col = false);
    template <typename T> void Pb(const T* bias, T* y);
    template <typename T> void Dx(const T* dy, const T* weights, T* dx);
    template <typename T> void Dw(const T* dy, const T* x, T *dw);
    template <typename T> void Db(const T* dy, T* db);

 private:
    template <typename T> void Im2Col(const T* im, T* col_buffer) {
        kernel::Im2Col<T, Context>(conv_in_channels, 
                     input_shape[0], input_shape[1],
                     kernel_size[0], kernel_size[1], 
                               stride[0], stride[1],
                                     pad[0], pad[1],
                           dilation[0], dilation[1], 
                                                 im,
                                        col_buffer);
    }
    template <typename T> void Col2Im(const T* col_buffer, T* im) {
        kernel::Col2Im<T, Context>(conv_in_channels, 
                     input_shape[0], input_shape[1],
                     kernel_size[0], kernel_size[1],  
                               stride[0], stride[1],
                                     pad[0], pad[1],
                           dilation[0], dilation[1],
                                         col_buffer,
                                                im);
    }
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_