#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
void ConvOpBase<Context>::ComputeOutputShape() {
    output_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (!ReverseDimensions()) {
            const TIndex idm = bottom_shape[spatial_axis + i];
            const TIndex dk = dilation[i] * (kernel_size[i] - 1) + 1;
            if (padding != "SAME") {
                const TIndex odm = (idm + 2 * pad[i] - dk) / stride[i] + 1;
                output_shape.push_back(odm);
            } else {
                TIndex odm = (idm + stride[i] - 1) / (float)stride[i];
                TIndex padding_needed = std::max(
                    TIndex(0), (odm - 1) * stride[i] + dk - idm);
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
                output_shape.push_back(odm);
            }
        } else {
            const TIndex idm = bottom_shape[spatial_axis + i];
            const TIndex dk = dilation[i] * (kernel_size[i] - 1) + 1;
            if (padding != "SAME") {
                const TIndex odm = stride[i] * (idm - 1) + dk - 2 * pad[i];
                output_shape.push_back(odm);
            } else {
                CHECK(output_dims_desc.size() > 0 ||
                         output_dims_value.size() > 0)
                    << "\nThe output shape must be specified "
                    << "if using SAME padding algorithm.";
                int given_ndim = (int)std::max(
                    output_dims_desc.size(), output_dims_value.size());
                CHECK_EQ(given_ndim, num_spatial_axes + 2)
                    << "\nThe len of output shape should be " 
                    << num_spatial_axes + 2
                    << ", but got " << output_dims_desc.size() << ".";
                TIndex odm = output_dims(spatial_axis + i);
                TIndex padding_needed = stride[i] * (idm - 1) + dk - odm;
                CHECK_GE(padding_needed, 0)
                    << "\nThe output shape is incorrect."
                    << "\nWith the given stride and kernel, "
                    << "dimension of axis " << spatial_axis + i
                    << " can be at most " << stride[i] * (idm - 1) + dk << ".";
                TIndex pad_l = padding_needed / 2;
                TIndex pad_r = padding_needed - pad_l;
                pad[i] = pad_l;
                output_shape.push_back(odm);
            }
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Wx(
    const T*                x,
    const T*                weights,
    T*                      y,
    bool                    skip_im2col) {
    auto* col_buffer = x;
    if (!is_1x1) {
        auto* workspace = ws()->template caches<T, Context>({ col_dim })[0];
        if (!skip_im2col) Im2Col(x, workspace);
        col_buffer = workspace;
    }
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                    conv_out_channels / group,
                         conv_out_spatial_dim,
                                   kernel_dim,
                1.0, weights + weight_offset * g,
                     col_buffer + col_offset * g,
                0.0, y + output_offset * g, ctx());
        } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasNoTrans,
                         conv_out_spatial_dim,
                    conv_out_channels / group,
                                   kernel_dim,
                1.0, col_buffer + col_offset * g,
                     weights + weight_offset * g,
                0.0, y + output_offset * g, ctx());
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Pb(const T* bias, T* y) {
    DECLARE_MULTIPLIER(multiplier, out_spatial_dim);
    if (data_format == "NCHW") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                num_output, out_spatial_dim, 1,
                    1.0, bias, multiplier,
                        1.0, y, ctx());
    } else if (data_format == "NHWC") {
        math::Gemm<T, Context>(
            CblasNoTrans, CblasNoTrans,
                out_spatial_dim, num_output, 1,
                    1.0, multiplier, bias,
                        1.0, y, ctx());
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dx(const T* dy, const T* weights, T* dx) {
    auto* col_buffer = is_1x1 ? dx :
        ws()->template caches<T, Context>({ col_dim })[0];
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm<T, Context>(
                CblasTrans, CblasNoTrans,
                                   kernel_dim,
                         conv_out_spatial_dim,
                    conv_out_channels / group,
                1.0, weights + weight_offset * g,
                          dy + output_offset * g,
                0.0, col_buffer + col_offset * g, ctx());
        } else if (data_format == "NHWC") {
             math::Gemm<T, Context>(
                 CblasNoTrans, CblasTrans,
                          conv_out_spatial_dim,
                                    kernel_dim,
                     conv_out_channels / group,
                 1.0, dy + output_offset * g,
                     weights + weight_offset * g,
                 0.0, col_buffer + col_offset * g, ctx());
        }
    }
    if (!is_1x1) Col2Im(col_buffer, dx);
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dw(const T* dy, const T* x, T *dw) {
    auto* col_buffer = x;
    if (!is_1x1) {
        auto* workspace = ws()->template caches<T, Context>({ col_dim })[0];
        Im2Col(x, workspace);
        col_buffer = workspace;
    }
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm<T, Context>(
                CblasNoTrans, CblasTrans,
                    conv_out_channels / group,
                                   kernel_dim,
                         conv_out_spatial_dim,
                1.0, dy + output_offset * g,
                    col_buffer + col_offset * g,
                1.0, dw + weight_offset * g, ctx());
        } else if (data_format == "NHWC") {
            math::Gemm<T, Context>(
                CblasTrans, CblasNoTrans,
                                   kernel_dim,
                    conv_out_channels / group,
                         conv_out_spatial_dim,
                1.0, col_buffer + col_offset * g,
                          dy + output_offset * g,
                1.0, dw + weight_offset * g, ctx());
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Db(const T* dy, T* db) {
    DECLARE_MULTIPLIER(multiplier, out_spatial_dim);
    if (data_format == "NCHW") {
        math::Gemv<T, Context>(
            CblasNoTrans, num_output, out_spatial_dim,
                1.0, dy, multiplier,
                    1.0, db, ctx());
    } else if (data_format == "NHWC") {
        math::Gemv<T, Context>(
            CblasTrans, out_spatial_dim, num_output,
                1.0, dy, multiplier,
                    1.0, db, ctx());
    }
}

template <class Context>
void ConvOpBase<Context>::Setup() {
    vector<int> ks = OperatorBase::Args<int>("kernel_size");
    for (int i = 0; i < num_spatial_axes; i++)
        kernel_size.push_back(i < ks.size() ? ks[i] : ks[0]);

    vector<int> s = OperatorBase::Args<int>("stride");
    for (int i = 0; i < num_spatial_axes; i++)
        stride.push_back(i < s.size() ? s[i] : s[0]);

    vector<int> p = OperatorBase::Args<int>("pad");
    for (int i = 0; i < num_spatial_axes; i++)
        pad.push_back(i < p.size() ? p[i] : p[0]);

    vector<int> d = OperatorBase::Args<int>("dilation");
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
    channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
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
    bias_shape = { num_output };

    //  determine the bottom and top shape
    bottom_shape = Input(0).dims();
    ComputeOutputShape();
    if (data_format == "NCHW") {
        top_shape.assign({ Input(0).dim(0), num_output });
        for (int i = 0; i < num_spatial_axes; i++)
            top_shape.push_back(output_shape[i]);
    } else if (data_format == "NHWC") {
        top_shape.assign({ Input(0).dim(0) });
        for (int i = 0; i < num_spatial_axes; i++)
            top_shape.push_back(output_shape[i]);
        top_shape.push_back(num_output);
    }
    Output(0)->Reshape(top_shape);

    //  determine the input shape for im2col/col2im
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(Output(0)->dim(spatial_axis + i));
        } else {
            input_shape.push_back(Input(0).dim(spatial_axis + i));
        }
    }

    //  determine the out spatial dim
    if (data_format == "NCHW") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = Input(0).count(spatial_axis);
        } else {
            conv_out_spatial_dim = Output(0)->count(spatial_axis);
        }
        out_spatial_dim = Output(0)->count(spatial_axis);
    } else if (data_format == "NHWC") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = Input(0).count(
                spatial_axis, (int)Input(0).ndim() - 1);
        } else {
            conv_out_spatial_dim = Output(0)->count(
                spatial_axis, (int)Output(0)->ndim() - 1);
        }
        out_spatial_dim = Output(0)->count(
            spatial_axis, (int)Output(0)->ndim() - 1);
    }

    //  determine the misc
    x_offset = Input(0).count(1);
    y_offset = Output(0)->count(1);
    kernel_dim = conv_in_channels / group * kernel_size[0] * kernel_size[1];
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    //  determine the workspace size for col buffer
    col_dim = kernel_dim * group;
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) 
            col_dim *= bottom_shape[spatial_axis + i];
        else col_dim *= output_shape[i];
    }
}

template <class Context>
void ConvOpBase<Context>::GradientReshape() {
    channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else{
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }
    //  determine the bottom and top shape
    bottom_shape = Input(0).dims();
    ComputeOutputShape();
    Output(0)->Reshape(bottom_shape);
    Output(1)->ReshapeLike(Input(1));
    Output(2)->Reshape({ num_output });

    //  determine the input shape for im2col/col2im
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(Input(-1).dim(spatial_axis + i));
        } else {
            input_shape.push_back(Input(0).dim(spatial_axis + i));
        }
    }

    //  determine the out spatial dim
    if (data_format == "NCHW") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = Input(0).count(spatial_axis);
        } else {
            conv_out_spatial_dim = Input(-1).count(spatial_axis);
        }
        out_spatial_dim = Input(-1).count(spatial_axis);
    } else if (data_format == "NHWC") {
        if (ReverseDimensions()) {
            conv_out_spatial_dim = Input(0).count(
                spatial_axis, (int)Input(0).ndim() - 1);
        } else {
            conv_out_spatial_dim = Input(-1).count(
                spatial_axis, (int)Input(-1).ndim() - 1);
        }
        out_spatial_dim = Input(-1).count(
            spatial_axis, (int)Input(-1).ndim() - 1);
    }

    //  determine the misc
    x_offset = Input(0).count(1);
    y_offset = Input(-1).count(1);
    kernel_dim = conv_in_channels / group * kernel_size[0] * kernel_size[1];
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    //  determine the workspace size for col buffer
    col_dim = kernel_dim * group;
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions())
            col_dim *= bottom_shape[spatial_axis + i];
        else col_dim *= output_shape[i];
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