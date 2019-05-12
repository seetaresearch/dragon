#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/conv_op_base.h"

namespace dragon {

#define DEFINE_SAME_PADDING(A, B) \
    A[i] = padding_needed / 2; \
    B[i] = padding_needed - A[i]

template <class Context>
void ConvOpBase<Context>::ComputeOutShape() {
    out_shape_.clear();
    for (int i = 0; i < num_axes_; i++) {
        if (!Transposed()) {
            auto idm = x_shape_[axis_ + i];
            auto dk = dilation_[i] * (kshape_[i] - 1) + 1;
            if (padding_.find("SAME") == string::npos) {
                // Explicit pads
                auto odm = (
                    idm + pad_l_[i] + pad_r_[i] - dk
                        ) / stride_[i] + 1;
                out_shape_.push_back(odm);
            } else {
                // Auto pads
                int64_t odm = (
                    idm + stride_[i] - 1
                        ) / (float)stride_[i];
                auto padding_needed = std::max(
                    int64_t(0), (odm - 1) * stride_[i] + dk - idm);
                out_shape_.push_back(odm);
                if (padding_ != "SAME_UPPER") {
                    DEFINE_SAME_PADDING(pad_l_, pad_r_);
                } else { 
                    DEFINE_SAME_PADDING(pad_r_, pad_l_);
                }  // SAME_LOWER or SAME
            }
        } else {
            auto idm = x_shape_[axis_ + i];
            auto dk = dilation_[i] * (kshape_[i] - 1) + 1;
            if (padding_.find("SAME") == string::npos) {
                // Explicit pads
                auto odm = stride_[i] * (idm - 1
                    ) + dk - pad_l_[i] - pad_r_[i];
                out_shape_.push_back(odm);
            } else {
                // Auto pads
                int output_shape_size = GET_ARGS_SIZE(output_shape);
                int output_padding_size = GET_ARGS_SIZE(output_padding);
                CHECK(output_shape_size == 0 ||
                      output_shape_size == num_axes_)
                    << "Excepted 0 or " << num_axes_
                    << " ints for output shape.";
                CHECK(output_padding_size == 0 ||
                      output_padding_size == num_axes_)
                    << "Excepted 0 or " << num_axes_
                    << " ints for output padding.";
                int64_t padding_needed, odm;
                if (output_padding_size) {
                    padding_needed = output_padding(i);
                    odm = stride_[i] * (idm - 1) + dk + padding_needed;
                } else if (output_shape_size) {
                    odm = output_shape(i);
                    padding_needed = odm - (stride_[i] * (idm - 1) + dk);
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
                out_shape_.push_back(odm);
                if (padding_ != "SAME_UPPER") {
                    DEFINE_SAME_PADDING(pad_l_, pad_r_);
                } else {
                    DEFINE_SAME_PADDING(pad_r_, pad_l_);
                }  // SAME_LOWER or SAME
            }
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Wx(
    const T*                x,
    const T*                w,
    T*                      y,
    bool                    skip) {
    auto* col = x;

    if (!is_1x1_) {
        auto* scratch = ws()->template
            data<T, Context>({ col_dim_ })[0];
        if (!skip) Im2Col(x, scratch);
        col = scratch;
    }

    for (int g = 0; g < group_; g++) {
        if (data_format() == "NCHW") {
            math::Gemm(
                CblasNoTrans,
                CblasNoTrans,
                conv_out_channels_ / group_,
                conv_out_dim_,
                kernel_dim_,
                1.f,
                w + w_ofs_ * g,
                col + col_ofs_ * g,
                0.f,
                y + output_ofs_ * g, ctx()
            );
        } else if (data_format() == "NHWC") {
            math::Gemm(
                CblasNoTrans,
                CblasTrans,
                conv_out_dim_,
                conv_out_channels_,
                kernel_dim_,
                1.f, col, w,
                0.f, y, ctx()
            );
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Pb(const T* bias, T* y) {
    DECLARE_MULTIPLIER(multiplier, out_dim_);
    kernel::BiasAdd(
        X(0).dim(0),
        num_output_,
        out_dim_,
        data_format(),
        bias, multiplier,
        y, ctx()
    );
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dx(const T* dy, const T* w, T* dx) {
    auto* col = is_1x1_ ? dx :
        ws()->template data
            <T, Context>({ col_dim_ })[0];
    for (int g = 0; g < group_; g++) {
        if (data_format() == "NCHW") {
            math::Gemm(
                CblasTrans,
                CblasNoTrans,
                kernel_dim_,
                conv_out_dim_,
                conv_out_channels_ / group_,
                1.f,
                w + w_ofs_ * g,
                dy + output_ofs_ * g,
                0.f,
                col + col_ofs_ * g, ctx()
            );
        } else if (data_format() == "NHWC") {
             math::Gemm(
                 CblasNoTrans,
                 CblasNoTrans,
                 conv_out_dim_,
                 kernel_dim_,
                 conv_out_channels_,
                 1.f, dy, w,
                 0.f, col, ctx()
             );
        }
    }
    if (!is_1x1_) Col2Im(col, dx);
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Dw(
    const T*                dy,
    const T*                x,
    T*                      dw,
    bool                    accum) {
    auto* col = x;

    if (!is_1x1_) {
        auto* scratch = ws()
            ->template data<T, Context>
                ({ col_dim_ })[0];
        Im2Col(x, scratch);
        col = scratch;
    }

    for (int g = 0; g < group_; g++) {
        if (data_format() == "NCHW") {
            math::Gemm(
                CblasNoTrans,
                CblasTrans,
                conv_out_channels_ / group_,
                kernel_dim_,
                conv_out_dim_,
                1.f,
                dy + output_ofs_ * g,
                col + col_ofs_ * g,
                accum ? 1.f : 0.f,
                dw + w_ofs_ * g, ctx()
            );
        } else if (data_format() == "NHWC") {
            math::Gemm(
                CblasTrans,
                CblasNoTrans,
                conv_out_channels_,
                kernel_dim_,
                conv_out_dim_,
                1.f,
                dy, col,
                accum ? 1.f : 0.f,
                dw, ctx()
            );
        }
    }
}

template <class Context> template <typename T>
void ConvOpBase<Context>::Db(const T* dy, T* db) {
    vec32_t dims, axes;
    if (data_format() == "NCHW") {
        dims = {
            (int)X(0).dim(0),
            (int)num_output_,
            (int)out_dim_,
        }, axes = { 0, 2 };
    } else if (data_format() == "NHWC") {
        dims = {
            (int)X(0).dim(0),
            (int)out_dim_,
            (int)num_output_,
        }, axes = { 0, 1 };
    }
    kernel::ReduceSum(
        3, dims.data(),
        2, axes.data(),
        1.f, dy,
        db, ctx()
    );
}

template <class Context>
void ConvOpBase<Context>::Setup(int num_axes) {
    num_axes_ = num_axes;

    auto at = [&](const vec64_t& vec, int i) {
        return i < vec.size() ? vec[i] : vec[0];
    };

    auto pads = OpArgs<int64_t>("pads");
    auto strides = OpArgs<int64_t>("strides");
    auto kshape = OpArgs<int64_t>("kernel_shape");
    auto dilations = OpArgs<int64_t>("dilations");

    for (int i = 0; i < num_axes; i++) {
        pad_l_.push_back(at(pads, i));
        stride_.push_back(at(strides, i));
        kshape_.push_back(at(kshape, i));
        dilation_.push_back(at(dilations, i));
    }

    if ((int64_t)pads.size() == (num_axes * 2)) {
        for (int i = 0; i < num_axes; i++)
            pad_r_.push_back(pads[num_axes + i]);
    } else {
        pad_r_.assign(pad_l_.begin(), pad_l_.end());
    }

    bool flag_1x1 = true;
    for (int i = 0; i < num_axes; i++) {
        flag_1x1 &= (
            pad_l_[i] == 0 &&
            pad_r_[i] == 0 &&
            stride_[i] == 1 &&
            kshape_[i] == 1
        );
        if (!flag_1x1) break;
    }
    is_1x1_ = flag_1x1 ? 1 : 0;
}

template <class Context>
void ConvOpBase<Context>::Reshape(bool backward) {
    auto* Y_ref = backward ? &X(-1) : Y(0);

    // Determine the in/out channels
    channels_ = data_format() == "NCHW" ?
                X(0).dim(1) : X(0).dim(-1);
    if (num_output_ <= 0) {
        // Infer the out channels from the weights shape
        num_output_ = X(1).count() / channels_;
        for (int i = 0; i < num_axes_; i++)
            num_output_ /= kshape_[i];
        CHECK_GT(num_output_, 0)
            << "\nFailed to infer the out channels "
            << "from weights: " << X(1).DimString();
    }
    if (Transposed()) {
        conv_out_channels_ = channels_;
        conv_in_channels_ = num_output_;
    } else {
        conv_out_channels_ = num_output_;
        conv_in_channels_ = channels_;
    }

    // Determine the weight and bias shape
    if (data_format() == "NCHW") {
        w_shape_ = {
            conv_out_channels_,
            conv_in_channels_ / group_
        };
        for (int i = 0; i < num_axes_; i++)
            w_shape_.push_back(kshape_[i]);
    } else if (data_format() == "NHWC") {
        w_shape_ = { conv_out_channels_ };
        for (int i = 0; i < num_axes_; i++)
            w_shape_.push_back(kshape_[i]);
        w_shape_.push_back(conv_in_channels_ / group_);
    }
    b_shape_ = { num_output_ };

    // Determine the Y shape
    x_shape_ = X(0).dims();
    ComputeOutShape();
    if (backward) {
        Y(0)->ReshapeLike(X(0));
        Y(1)->ReshapeLike(X(1));
        Y(2)->Reshape({ num_output_ });
    } else {
        if (data_format() == "NCHW") {
            y_shape_ = { X(0).dim(0), num_output_ };
            for (int i = 0; i < num_axes_; i++)
                y_shape_.push_back(out_shape_[i]);
        } else if (data_format() == "NHWC") {
            y_shape_ = { X(0).dim(0) };
            for (int i = 0; i < num_axes_; i++)
                y_shape_.push_back(out_shape_[i]);
            y_shape_.push_back(num_output_);
        }
        Y(0)->Reshape(y_shape_);
    }

    // Determine the input shape for im2col/col2im
    in_shape_.clear();
    for (int i = 0; i < num_axes_; i++) {
        if (Transposed()) {
            in_shape_.push_back(Y_ref->dim(axis_ + i));
        } else {
            in_shape_.push_back(X(0).dim(axis_ + i));
        }
    }

    // Determine the out spatial dim
    auto end_axis = X(0).ndim() - 1;
    if (data_format() == "NCHW") {
        if (Transposed()) {
            conv_out_dim_ = X(0).count(axis_);
        } else {
            conv_out_dim_ = Y_ref->count(axis_);
        }
        out_dim_ = Y_ref->count(axis_);
    } else if (data_format() == "NHWC") {
        if (Transposed()) {       
            conv_out_dim_ = X(0).count(axis_, end_axis);
        } else {
            conv_out_dim_ = Y_ref->count(axis_, end_axis);
        }
        out_dim_ = Y_ref->count(axis_, end_axis);
    }

    // Determine the misc
    x_ofs_ = X(0).count(1);
    y_ofs_ = Y_ref->count(1);
    kernel_dim_ = conv_in_channels_ / group_;
    for (int i = 0; i < num_axes_; i++) kernel_dim_ *= kshape_[i];
    col_ofs_ = kernel_dim_ * conv_out_dim_;
    w_ofs_ = conv_out_channels_ * kernel_dim_ / group_;
    output_ofs_ = conv_out_channels_ * conv_out_dim_ / group_;

    // Determine the workspace size for col buffer
    col_dim_ = kernel_dim_ * group_;
    for (int i = 0; i < num_axes_; i++) {
        if (Transposed()) {
            col_dim_ *= x_shape_[axis_ + i];
        } else {
            col_dim_ *= out_shape_[i];
        }
    }
}

template class ConvOpBase<CPUContext>;
template void ConvOpBase<CPUContext>
    ::Wx(const float*, const float*, float*, bool);
template void ConvOpBase<CPUContext>
    ::Pb(const float*, float*);
template void ConvOpBase<CPUContext>
    ::Dx(const float*, const float*, float*);
template void ConvOpBase<CPUContext>
    ::Dw(const float*, const float*, float*, bool);
template void ConvOpBase<CPUContext>
    ::Db(const float*, float*);

#ifdef WITH_CUDA
template class ConvOpBase<CUDAContext>;
template void ConvOpBase<CUDAContext>
    ::Wx(const float*, const float*, float*, bool);
template void ConvOpBase<CUDAContext>
    ::Pb(const float*, float*);
template void ConvOpBase<CUDAContext>
    ::Dx(const float*, const float*, float*);
template void ConvOpBase<CUDAContext>
    ::Dw(const float*, const float*, float*, bool);
template void ConvOpBase<CUDAContext>
    ::Db(const float*, float*);
#endif

#undef DEFINE_SAME_PADDING

}  // namespace dragon