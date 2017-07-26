#include "operators/vision/conv_op.h"
#include "core/workspace.h"
#include "utils/filler.h"

namespace dragon {

template <class Context> template <typename T>
void ConvOpBase<Context>::Wx(const T* x, const T* weights, T* y, bool skip_im2col) {
    const T* col_buff_ = x;
    if (!is_1x1) {
        if (!skip_im2col) Im2Col(x, col_buffer->template mutable_data<T, Context>());
        col_buff_ = col_buffer->data<T, Context>();
    }
    for (int g = 0; g < group; g++) {
        math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, 
                                conv_out_channels / group,
                                     conv_out_spatial_dim, 
                                               kernel_dim, 
                         1.0, weights + weight_offset * g, 
                               col_buff_ + col_offset * g, 
                              0.0, y + output_offset * g);
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Pb(const T* bias, T* y) {
    math::Gemm<T, Context>(CblasNoTrans, CblasNoTrans, 
                                           num_output, 
                                      out_spatial_dim, 
                                                    1, 
                                            1.0, bias, 
         bias_multiplier->template data<T, Context>(), 
                                              1.0, y);
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dx(const T* dy, const T* weights, T* dx) {
    T* col_buff_ = col_buffer->template mutable_data<T, Context>();
    if (is_1x1) col_buff_ = dx;
    for (int g = 0; g < group; g++) {
        math::Gemm<T, Context>(CblasTrans, CblasNoTrans, 
                                             kernel_dim, 
                                   conv_out_spatial_dim, 
                              conv_out_channels / group, 
                       1.0, weights + weight_offset * g, 
                                 dy + output_offset * g, 
                        0.0, col_buff_ + col_offset * g);
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
        math::Gemm<T, Context>(CblasNoTrans, CblasTrans,
                              conv_out_channels / group,
                                             kernel_dim, 
                                   conv_out_spatial_dim, 
                            1.0, dy + output_offset * g,
                             col_buff_ + col_offset * g, 
                            1.0, dw + weight_offset * g);
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Db(const T* dy, T* db) {
    math::Gemv<T, Context>(CblasNoTrans, num_output, out_spatial_dim, 
                                                             1.0, dy, 
                        bias_multiplier->template data<T, Context>(), 
                                                            1.0, db);
}

template <class Context>
void ConvOpBase<Context>::Reshape() {
    channels = input(0).dim(channel_axis);
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else {
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }
    weight_shape.assign({ conv_out_channels,
                          conv_in_channels / group,
                          kernel_size[0],
                          kernel_size[1]});
    bias_shape.assign(1, num_output);

    //  compute bottom and top shape
    bottom_shape = input(0).dims();
    ComputeOutputShape();
    vector<TIndex> top_shape({input(0).dim(0), 
                              num_output, 
                              output_shape[0], 
                              output_shape[1]});
    output(0)->Reshape(top_shape);

    if (ReverseDimensions()) {
        conv_out_spatial_dim = input(0).count(channel_axis + 1);
    } else {
        conv_out_spatial_dim = output(0)->count(channel_axis + 1);
    }

    //  compute input shape
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(output(0)->dim(channel_axis + i + 1));
        } else {
            input_shape.push_back(input(0).dim(channel_axis + i + 1));
        }
    }

    kernel_dim = conv_in_channels / group * kernel_size[0] * kernel_size[1];
    out_spatial_dim = output(0)->count(channel_axis + 1);

    x_offset = input(0).count(channel_axis);
    y_offset = output(0)->count(channel_axis);
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    //    compute col buffer shape
    col_buffer_shape.clear();
    col_buffer_shape.push_back(kernel_dim * group);
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            col_buffer_shape.push_back(bottom_shape[channel_axis + i + 1]);
        } else {
            col_buffer_shape.push_back(output_shape[i]);
        }
    }
}

template <class Context>
void ConvOpBase<Context>::GradientReshape() {
    channels = input(0).dim(channel_axis);
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else{
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }
    bottom_shape = input(0).dims();
    ComputeOutputShape();
    output(0)->Reshape(bottom_shape);
    output(1)->ReshapeLike(input(1));
    output(2)->Reshape(vector<TIndex>(1, num_output));

    if (ReverseDimensions()) {
        conv_out_spatial_dim = input(0).count(channel_axis + 1);
    } else {
        conv_out_spatial_dim = input(2).count(channel_axis + 1);
    }

    //    compute input shape
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++){
        if (ReverseDimensions()) {
            input_shape.push_back(input(2).dim(channel_axis + i + 1));
        } else {
            input_shape.push_back(input(0).dim(channel_axis + i + 1));
        }
    }

    kernel_dim = input(1).count(1);    //    in * kh * kw
    out_spatial_dim = input(2).count(channel_axis + 1);

    x_offset = input(0).count(channel_axis);
    y_offset = input(2).count(channel_axis);
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    //    compute col buffer shape
    col_buffer_shape.clear();
    col_buffer_shape.push_back(kernel_dim * group);
    for (int i = 0; i < num_spatial_axes; i++){
        if (ReverseDimensions()) {
            col_buffer_shape.push_back(bottom_shape[channel_axis + i + 1]);
        } else {
            col_buffer_shape.push_back(output_shape[i]);
        }
    }
}

template class ConvOpBase<CPUContext>;
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
