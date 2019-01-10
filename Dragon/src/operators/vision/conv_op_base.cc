#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/conv_op_base.h"

namespace dragon {

#define DEFINE_SAME_PADDING(A, B) \
    A[i] = padding_needed / 2; \
    B[i] = padding_needed - A[i]

template <class Context>
void ConvOpBase<Context>::ComputeOutputShape() {
    output_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (!ReverseDimensions()) {
            const int64_t idm = bottom_shape[spatial_axis + i];
            const int64_t dk = dilation[i] * (kernel_shape[i] - 1) + 1;
            if (padding.find("SAME") == string::npos) {
                // Explicit pads
                const int64_t odm = (idm + pad_l[i] + pad_r[i] - dk) / stride[i] + 1;
                output_shape.push_back(odm);
            } else {
                // Auto pads
                int64_t odm = (idm + stride[i] - 1) / (float)stride[i];
                int64_t padding_needed = std::max(
                    (int64_t)0, (odm - 1) * stride[i] + dk - idm);
                output_shape.push_back(odm);
                if (padding != "SAME_UPPER") { DEFINE_SAME_PADDING(pad_l, pad_r); }
                else { DEFINE_SAME_PADDING(pad_r, pad_l); }  // SAME_LOWER or SAME
            }
        } else {
            const int64_t idm = bottom_shape[spatial_axis + i];
            const int64_t dk = dilation[i] * (kernel_shape[i] - 1) + 1;
            if (padding.find("SAME") == string::npos) {
                // Explicit pads
                const int64_t odm = stride[i] * (idm - 1) + dk - pad_l[i] - pad_r[i];
                output_shape.push_back(odm);
            } else {
                // Auto pads
                int given_num_output_shape = (int)std::max(
                    output_shape_spec_desc.size(),
                        output_shape_spec_value.size());
                int given_num_output_padding = (int)std::max(
                    output_padding_desc.size(),
                        output_padding_value.size());
                CHECK(given_num_output_shape == 0 ||
                    given_num_output_shape == num_spatial_axes)
                    << "Excepted 0 or " << num_spatial_axes
                    << " ints for output shape.";
                CHECK(given_num_output_padding == 0 ||
                      given_num_output_padding == num_spatial_axes)
                    << "Excepted 0 or " << num_spatial_axes
                    << " ints for output padding.";
                int64_t padding_needed, odm;
                if (given_num_output_padding) {
                    padding_needed = output_padding(i);
                    odm = stride[i] * (idm - 1) + dk + padding_needed;
                } else if (given_num_output_shape) {
                    odm = output_shape_spec(i);
                    padding_needed = odm - (stride[i] * (idm - 1) + dk);
                    CHECK_GE(padding_needed, 0)
                        << "\nThe output shape is incorrect."
                        << "\nWith the given stride and kernel, "
                        << "dimension of spatial axis " << i
                        << " should be at least "
                        << odm - padding_needed << ".";
                } else {
                    LOG(FATAL)
                        << "Excepted the output padding or output shape"
                           "for \"SAME\" padding algorithm.";
                }
                output_shape.push_back(odm);
                if (padding != "SAME_UPPER") { DEFINE_SAME_PADDING(pad_l, pad_r); }
                else { DEFINE_SAME_PADDING(pad_r, pad_l); }  // SAME_LOWER or SAME
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
        auto* WSdata = ws()->template
            caches<T, Context>({ col_dim })[0];
        if (!skip_im2col) Im2Col(x, WSdata);
        col_buffer = WSdata;
    }

    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm(
                CblasNoTrans, CblasNoTrans,
                    conv_out_channels / group,
                         conv_out_spatial_dim,
                                   kernel_dim,
                1.f, weights + weight_offset * g,
                     col_buffer + col_offset * g,
                0.f, y + output_offset * g, ctx());
        } else if (data_format == "NHWC") {
            math::Gemm(
                CblasNoTrans, CblasTrans,
                    conv_out_spatial_dim, conv_out_channels,
                        kernel_dim,
                1.f, col_buffer, weights, 0.f, y, ctx());
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Pb(const T* bias, T* y) {
    DECLARE_MULTIPLIER(multiplier, out_spatial_dim);
    kernel::BiasAdd(Output(0)->count(),
        Input(0).dim(0), num_output, out_spatial_dim,
            data_format, bias, multiplier, y, ctx());
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dx(const T* dy, const T* weights, T* dx) {
    auto* col_buffer = is_1x1 ? dx :
        ws()->template caches<T, Context>({ col_dim })[0];
    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm(
                CblasTrans, CblasNoTrans,
                    kernel_dim, conv_out_spatial_dim,
                        conv_out_channels / group,
                1.f, weights + weight_offset * g,
                          dy + output_offset * g,
                0.f, col_buffer + col_offset * g, ctx());
        } else if (data_format == "NHWC") {
             math::Gemm(
                 CblasNoTrans, CblasNoTrans,
                     conv_out_spatial_dim, kernel_dim,
                         conv_out_channels,
                 1.f, dy, weights, 0.f, col_buffer, ctx());
        }
    }
    if (!is_1x1) Col2Im(col_buffer, dx);
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dw(const T* dy, const T* x, T *dw) {
    auto* col_buffer = x;

    if (!is_1x1) {
        auto* WSdata = ws()->template
            caches<T, Context>({ col_dim })[0];
        Im2Col(x, WSdata);
        col_buffer = WSdata;
    }

    for (int g = 0; g < group; g++) {
        if (data_format == "NCHW") {
            math::Gemm(
                CblasNoTrans, CblasTrans,
                    conv_out_channels / group,
                                   kernel_dim,
                         conv_out_spatial_dim,
                1.f, dy + output_offset * g,
                    col_buffer + col_offset * g,
                1.f, dw + weight_offset * g, ctx());
        } else if (data_format == "NHWC") {
            math::Gemm(
                CblasTrans, CblasNoTrans,
                    conv_out_channels, kernel_dim,
                        conv_out_spatial_dim,
                1.f, dy, col_buffer, 1.f, dw, ctx());
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Db(const T* dy, T* db) {
    DECLARE_MULTIPLIER(multiplier, out_spatial_dim);
    if (data_format == "NCHW") {
        math::Gemv(
            CblasNoTrans, num_output, out_spatial_dim,
                1.f, dy, multiplier,
                    1.f, db, ctx());
    } else if (data_format == "NHWC") {
        math::Gemv(
            CblasTrans, out_spatial_dim, num_output,
                1.f, dy, multiplier,
                    1.f, db, ctx());
    }
}

template <class Context>
void ConvOpBase<Context>::Setup() {
    auto ks = OperatorBase::Args<int64_t>("kernel_shape");
    for (int i = 0; i < num_spatial_axes; i++)
        kernel_shape.emplace_back(i < ks.size() ? ks[i] : ks[0]);

    auto s = OperatorBase::Args<int64_t>("strides");
    for (int i = 0; i < num_spatial_axes; i++)
        stride.emplace_back(i < s.size() ? s[i] : s[0]);

    auto p = OperatorBase::Args<int64_t>("pads");
    for (int i = 0; i < num_spatial_axes; i++)
        pad_l.push_back(i < p.size() ? p[i] : p[0]);

    if ((int64_t)p.size() == (num_spatial_axes * 2)) {
        for (int i = 0; i < num_spatial_axes; i++)
            pad_r.push_back(p[num_spatial_axes + i]);
    } else {
        pad_r.assign(pad_l.begin(), pad_l.end());
    }

    auto d = OperatorBase::Args<int64_t>("dilations");
    for (int i = 0; i < num_spatial_axes; i++)
        dilation.push_back(i < d.size() ? d[i] : d[0]);

    is_1x1 = true;
    for (int i = 0; i < num_spatial_axes; i++) {
        is_1x1 &= (kernel_shape[i] == 1 && stride[i] == 1 &&
                   pad_l[i] == 0 && pad_r[i] == 0);
        if (!is_1x1) break;
    }
}

template <class Context>
void ConvOpBase<Context>::Reshape() {
    // Determine the in/out channels
    channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
    if (num_output <= 0) {
        // Infer the out channels from the weights shape
        num_output = Input(1).count() / channels;
        for (int i = 0; i < num_spatial_axes; i++)
            num_output /= kernel_shape[i];
        CHECK_GT(num_output, 0)
            << "\nFailed to infer the out channels from"
            << "the weights shape: " << Input(1).DimString();
    }
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else {
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }

    // Determine the weight and bias shape
    if (data_format == "NCHW") {
        weight_shape.assign({ conv_out_channels,
                              conv_in_channels / group });
        for (int i = 0; i < num_spatial_axes; i++)
            weight_shape.push_back(kernel_shape[i]);
    } else if (data_format == "NHWC") {
        weight_shape.assign({ conv_out_channels });
        for (int i = 0; i < num_spatial_axes; i++)
            weight_shape.push_back(kernel_shape[i]);
        weight_shape.push_back(conv_in_channels / group);
    }
    bias_shape = { num_output };

    // Determine the bottom and top shape
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

    // Determine the input shape for im2col/col2im
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(Output(0)->dim(spatial_axis + i));
        } else {
            input_shape.push_back(Input(0).dim(spatial_axis + i));
        }
    }

    // Determine the out spatial dim
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
                spatial_axis, Input(0).ndim() - 1);
        } else {
            conv_out_spatial_dim = Output(0)->count(
                spatial_axis, Output(0)->ndim() - 1);
        }
        out_spatial_dim = Output(0)->count(
            spatial_axis, Output(0)->ndim() - 1);
    }

    // Determine the misc
    x_offset = Input(0).count(1);
    y_offset = Output(0)->count(1);
    kernel_dim = conv_in_channels / group;
    for (int i = 0; i < num_spatial_axes; i++) kernel_dim *= kernel_shape[i];
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    // Determine the workspace size for col buffer
    col_dim = kernel_dim * group;
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) 
            col_dim *= bottom_shape[spatial_axis + i];
        else col_dim *= output_shape[i];
    }
}

template <class Context>
void ConvOpBase<Context>::GradientReshape() {
    // Determine the in/out channels
    channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
    if (num_output <= 0) {
        // Infer the out channels from the weights shape
        num_output = Input(1).count() / channels;
        for (int i = 0; i < num_spatial_axes; i++)
            num_output /= kernel_shape[i];
        CHECK_GT(num_output, 0)
            << "\nFailed to infer the out channels from"
            << "the weights shape: " << Input(1).DimString();
    }
    if (ReverseDimensions()) {
        conv_out_channels = channels;
        conv_in_channels = num_output;
    } else{
        conv_out_channels = num_output;
        conv_in_channels = channels;
    }

    // Determine the bottom and top shape
    bottom_shape = Input(0).dims();
    ComputeOutputShape();
    Output(0)->Reshape(bottom_shape);
    Output(1)->ReshapeLike(Input(1));
    Output(2)->Reshape({ num_output });

    // Determine the input shape for im2col/col2im
    input_shape.clear();
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions()) {
            input_shape.push_back(Input(-1).dim(spatial_axis + i));
        } else {
            input_shape.push_back(Input(0).dim(spatial_axis + i));
        }
    }

    // Determine the out spatial dim
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
                spatial_axis, Input(0).ndim() - 1);
        } else {
            conv_out_spatial_dim = Input(-1).count(
                spatial_axis, Input(-1).ndim() - 1);
        }
        out_spatial_dim = Input(-1).count(
            spatial_axis, Input(-1).ndim() - 1);
    }

    // Determine the misc
    x_offset = Input(0).count(1);
    y_offset = Input(-1).count(1);
    kernel_dim = conv_in_channels / group;
    for (int i = 0; i < num_spatial_axes; i++) kernel_dim *= kernel_shape[i];
    weight_offset = conv_out_channels * kernel_dim / group;
    col_offset = kernel_dim * conv_out_spatial_dim;
    output_offset = conv_out_channels * conv_out_spatial_dim / group;

    // Determine the workspace size for col buffer
    col_dim = kernel_dim * group;
    for (int i = 0; i < num_spatial_axes; i++) {
        if (ReverseDimensions())
            col_dim *= bottom_shape[spatial_axis + i];
        else col_dim *= output_shape[i];
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

#undef DEFINE_SAME_PADDING

}  // namespace dragon