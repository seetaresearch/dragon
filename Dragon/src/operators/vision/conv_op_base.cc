#include "operators/vision/conv_op_base.h"
#include "core/workspace.h"
#include "utils/filler.h"

namespace dragon {

template <class Context>
void ConvOpBase<Context>::ComputeOutputShape() {
    output_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (!ReverseDimensions()) {
            const TIndex input_dim = bottom_shape[spatial_axis + i];
            const TIndex dilated_kernel = dilation[i] * (kernel_size[i] - 1) + 1;
            if (padding != "SAME") {
                const TIndex output_dim = (input_dim + 2 * pad[i] - dilated_kernel) / stride[i] + 1;
                output_shape.push_back(output_dim);
            } else {
                TIndex output_dim = (input_dim + stride[i] - 1) / (float)stride[i];
                TIndex padding_needed = std::max(TIndex(0), (output_dim - 1) * stride[i] + dilated_kernel - input_dim);
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
                output_shape.push_back(output_dim);
            }
        } else {
            const TIndex input_dim = bottom_shape[spatial_axis + i];
            const TIndex dilated_kernel = dilation[i] * (kernel_size[i] - 1) + 1;
            if (padding != "SAME") {
                const TIndex output_dim = stride[i] * (input_dim - 1) + dilated_kernel - 2 * pad[i];
                output_shape.push_back(output_dim);
            } else {
                CHECK(output_dims_desc.size() > 0 || output_dims_value.size() > 0) 
                    << "\nThe output shape must be specified if using SAME padding algorithm.";
                int given_ndim = (int)std::max(output_dims_desc.size(), output_dims_value.size());
                CHECK_EQ(given_ndim, num_spatial_axes + 2)
                    << "\nThe len of output shape should be " << num_spatial_axes + 2
                    << ", but got " << output_dims_desc.size() << "."; 
                TIndex output_dim = output_dims(spatial_axis + i);
                TIndex padding_needed = stride[i] * (input_dim - 1) + dilated_kernel - output_dim;
                CHECK_GE(padding_needed, 0)
                    << "\nThe output shape is incorrect."
                    << "\nWith the given stride and kernel, dimension of axis " << spatial_axis + i
                    << " can be at most " << stride[i] * (input_dim - 1) + dilated_kernel << ".";
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
                output_shape.push_back(output_dim);
            }
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Wx(const T* x, const T* weights, T* y, bool skip_im2col) {
    const T* col_buff_ = x;
    if (!is_1x1) {
        if (!skip_im2col) Im2Col(x, col_buffer->template mutable_data<T, Context>());
        col_buff_ = col_buffer->data<T, Context>();
    }
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") { 
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans,
                                    conv_out_channels / group,
                                         conv_out_spatial_dim,
                                                   kernel_dim,
                             1.0, weights + weight_offset * g,
                                   col_buff_ + col_offset * g,
                                  0.0, y + output_offset * g);
        } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans,
                                         conv_out_spatial_dim,
                                    conv_out_channels / group,
                                                   kernel_dim,
                              1.0, col_buff_ + col_offset * g,
                                  weights + weight_offset * g,
                                  0.0, y + output_offset * g);
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Pb(const T* bias, T* y) {
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans,
                                               num_output,
                                          out_spatial_dim,
                                                        1,
                                                1.0, bias,
             bias_multiplier->template data<T, Context>(),
                                                  1.0, y);
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans,
                                          out_spatial_dim,
                                               num_output,
                                                        1,
        1.0, bias_multiplier->template data<T, Context>(),
                                                     bias,
                                                  1.0, y);
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dx(const T* dy, const T* weights, T* dx) {
    T* col_buff_ = col_buffer->template mutable_data<T, Context>();
    if (is_1x1) col_buff_ = dx;
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm<T, Context>(CblasTrans, CblasNoTrans,
                                                 kernel_dim,
                                       conv_out_spatial_dim,
                                  conv_out_channels / group,
                           1.0, weights + weight_offset * g,
                                     dy + output_offset * g,
                           0.0, col_buff_ + col_offset * g);
        } else if (data_format == "NHWC") {
             math::Gemm<T, Context>(CblasNoTrans, CblasTrans,
                                        conv_out_spatial_dim,
                                                  kernel_dim,
                                   conv_out_channels / group,
                                 1.0, dy + output_offset * g,
                                 weights + weight_offset * g,
                            0.0, col_buff_ + col_offset * g);
        }
    }
    if (!is_1x1) Col2Im(col_buff_, dx);
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dw(const T* dy, const T* x, T *dw) {
    const T *col_buff_ = x;
    if (!is_1x1) {
        Im2Col(x, col_buffer->template mutable_data<T, Context>());
        col_buff_ = col_buffer->template data<T, Context>();
    }
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") { 
            math::Gemm<T, Context>(CblasNoTrans, CblasTrans,
                                  conv_out_channels / group,
                                                 kernel_dim,
                                       conv_out_spatial_dim,
                                1.0, dy + output_offset * g,
                                 col_buff_ + col_offset * g,
                               1.0, dw + weight_offset * g);
        } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(CblasTrans, CblasNoTrans,
                                                 kernel_dim,
                                  conv_out_channels / group,
                                       conv_out_spatial_dim,
                            1.0, col_buff_ + col_offset * g,
                                     dy + output_offset * g,
                               1.0, dw + weight_offset * g);
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Db(const T* dy, T* db) {
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(CblasNoTrans, num_output, out_spatial_dim,
                                                                 1.0, dy,
                            bias_multiplier->template data<T, Context>(),
                                                                1.0, db);
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(CblasTrans, out_spatial_dim, num_output,
                                                               1.0, dy,
                          bias_multiplier->template data<T, Context>(),
                                                              1.0, db);
    }
}

template <class Context>
void ConvOpBase<Context>::Setup() {
    vector<int> ks = OperatorBase::GetRepeatedArg<int>("kernel_size");
    for (int i = 0; i < num_spatial_axes; i++)
        kernel_size.push_back(i < ks.size() ? ks[i] : ks[0]);

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

template <class Context>
void ConvOpBase<Context>::Reshape() {
    channels = data_format == "NCHW" ? input(0).dim(1) : input(0).dim(-1);
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else {
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }
    //  determine the weight and bias shape
    if (data_format == "NCHW") {
        weight_shape.assign({ conv_out_channels,
                              conv_in_channels / group });
        for (int i = 0; i < num_spatial_axes; i++)
            weight_shape.push_back(kernel_size[i]);
    } else if (data_format == "NHWC") {
        weight_shape.clear();
        for (int i = 0; i < num_spatial_axes; i++)
            weight_shape.push_back(kernel_size[i]);
        weight_shape.push_back(conv_in_channels / group);
        weight_shape.push_back(conv_out_channels);
    }
    bias_shape.assign(1, num_output);

    //  determine the bottom and top shape
    bottom_shape = input(0).dims();
    ComputeOutputShape();
    if (data_format == "NCHW") {
        top_shape.assign({ input(0).dim(0), num_output });
        for (int i = 0; i < num_spatial_axes; i++)
            top_shape.push_back(output_shape[i]);
    } else if (data_format == "NHWC") {
        top_shape.assign({ input(0).dim(0) });
        for (int i = 0; i < num_spatial_axes; i++)
            top_shape.push_back(output_shape[i]);
        top_shape.push_back(num_output);
    }
    output(0)->Reshape(top_shape);

    //  determine the input shape for im2col/col2im
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(output(0)->dim(spatial_axis + i));
        } else {
            input_shape.push_back(input(0).dim(spatial_axis + i));
        }
    }

    //  determine the out spatial dim
    if (data_format == "NCHW") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = input(0).count(spatial_axis);
        } else {
            conv_out_spatial_dim = output(0)->count(spatial_axis);
        }
        out_spatial_dim = output(0)->count(spatial_axis);
    } else if (data_format == "NHWC") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = input(0).count(spatial_axis, (int)input(0).ndim() - 1);
        } else {
            conv_out_spatial_dim = output(0)->count(spatial_axis, (int)output(0)->ndim() - 1);
        }
        out_spatial_dim = output(0)->count(spatial_axis, (int)output(0)->ndim() - 1);
    }

    //  determine the misc
    x_offset = input(0).count(1);
    y_offset = output(0)->count(1);
    kernel_dim = conv_in_channels / group * kernel_size[0] * kernel_size[1];
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    //  determine the col shape
    col_shape.clear();
    if (data_format == "NCHW") {
        col_shape.push_back(kernel_dim * group);
        for (int i = 0; i < num_spatial_axes; i++) {
            if (ReverseDimensions()) col_shape.push_back(bottom_shape[spatial_axis + i]);
            else col_shape.push_back(output_shape[i]);
        }
    } else if (data_format == "NHWC") {
        for (int i = 0; i < num_spatial_axes; i++) {
            if (ReverseDimensions()) col_shape.push_back(bottom_shape[spatial_axis + i]);
            else col_shape.push_back(output_shape[i]);
        }
        col_shape.push_back(kernel_dim * group);
    }
}

template <class Context>
void ConvOpBase<Context>::GradientReshape() {
    channels = data_format == "NCHW" ? input(0).dim(1) : input(0).dim(-1);
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else{
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }
    //  determine the bottom and top shape
    bottom_shape = input(0).dims();
    ComputeOutputShape();
    output(0)->Reshape(bottom_shape);
    output(1)->ReshapeLike(input(1));
    output(2)->Reshape(vector<TIndex>(1, num_output));

    //  determine the input shape for im2col/col2im
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(input(-1).dim(spatial_axis + i));
        } else {
            input_shape.push_back(input(0).dim(spatial_axis + i));
        }
    }

    //  determine the out spatial dim
    if (data_format == "NCHW") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = input(0).count(spatial_axis);
        } else {
            conv_out_spatial_dim = input(-1).count(spatial_axis);
        }
        out_spatial_dim = input(-1).count(spatial_axis);
    } else if (data_format == "NHWC") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = input(0).count(spatial_axis, (int)input(0).ndim() - 1);
        } else {
            conv_out_spatial_dim = input(-1).count(spatial_axis, (int)input(-1).ndim() - 1);
        }
        out_spatial_dim = input(-1).count(spatial_axis, (int)input(-1).ndim() - 1);
    }

    //  determine the misc
    x_offset = input(0).count(1);
    y_offset = input(-1).count(1);
    kernel_dim = conv_in_channels / group * kernel_size[0] * kernel_size[1];
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    //  determine the col shape
    col_shape.clear();
    if (data_format == "NCHW") {
        col_shape.push_back(kernel_dim * group);
        for (int i = 0; i < num_spatial_axes; i++) {
            if (ReverseDimensions()) col_shape.push_back(bottom_shape[spatial_axis + i]);
            else col_shape.push_back(output_shape[i]);
        }
    } else if (data_format == "NHWC") {
        for (int i = 0; i < num_spatial_axes; i++) {
            if (ReverseDimensions()) col_shape.push_back(bottom_shape[spatial_axis + i]);
            else col_shape.push_back(output_shape[i]);
        }
        col_shape.push_back(kernel_dim * group);
    }
}

template class ConvOpBase<CPUContext>;;
template void ConvOpBase<CPUContext>::Wx(const float*, const float*, float*, bool);
template void ConvOpBase<CPUContext>::Pb(const float*, float*);
template void ConvOpBase<CPUContext>::Dx(const float*, const float*, float*);
template void ConvOpBase<CPUContext>::Dw(const float*, const float*, float*);
template void ConvOpBase<CPUContext>::Db(const float*, float*);

#ifdef WITH_CUDA
template class ConvOpBase<CUDAContext>;
template void ConvOpBase<CUDAContext>::Wx(const float*, const float*, float*, bool);
template void ConvOpBase<CUDAContext>::Pb(const float*, float*);
template void ConvOpBase<CUDAContext>::Dx(const float*, const float*, float*);
template void ConvOpBase<CUDAContext>::Dw(const float*, const float*, float*);
template void ConvOpBase<CUDAContext>::Db(const float*, float*);
#endif

}    // namespace dragon