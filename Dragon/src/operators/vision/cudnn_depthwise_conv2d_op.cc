#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/depthwise_conv_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNDepthwiseConv2dOp<Context>::RunImpl() {
    TENSOR_FILL(X(1), w_shape_);
    if (HasBias()) {
        TENSOR_FILL(X(2), b_shape_);
        if (data_format() == "NCHW") {
            CuDNNSetTensor4dDesc<T>(
                &bias_desc_, data_format(),
                vec64_t({ 1, num_output_, 1, 1 })
            );
        } else if (data_format() == "NHWC") {
            CuDNNSetTensor4dDesc<T>(
                &bias_desc_, data_format(),
                vec64_t({ 1, 1, 1, num_output_ })
            );
        }
    }

    CuDNNSetTensor4dDesc<T>(
        &output_desc_,
        data_format(),
        Y(0)->dims()
    );

    auto* x = X(0).template data<T, Context>();
    auto* w = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::DepthwiseConv2d(
        X(0).dim(0), channels_,
        in_shape_[0], in_shape_[1],
        out_shape_[0], out_shape_[1],
        kshape_[0], kshape_[1],
        stride_[0], stride_[1],
        pad_l_[0], pad_l_[1],
        dilation_[0], dilation_[1],
        data_format(),
        x, w,
        y, ctx()
    );

    if (HasBias()) {
        auto* b = X(2).template data<T, Context>();
        CUDNN_CHECK(cudnnAddTensor(
            ctx()->cudnn_handle(),
            CuDNNType<T>::one,
            bias_desc_, b,
            CuDNNType<T>::one,
            output_desc_, y
        ));
    }
}

template <class Context>
void CuDNNDepthwiseConv2dOp<Context>::RunOnDevice() {
    group_ = channels_ = data_format()
        == "NCHW" ? X(0).dim(1) : X(0).dim(-1);
    CHECK_EQ(channels_, num_output_)
        << "\nExcepted in/out channels unchanged.";
    ConvOpBase<Context>::Reshape();

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void CuDNNDepthwiseConv2dGradientOp<Context>::RunImpl() {
    if (Y(2)->name() != "NULL") {
        CuDNNSetTensor4dDesc<T>(
            &input_desc_,
            data_format(),
            X(-1).dims()
        );
        if (data_format() == "NCHW") {
            CuDNNSetTensor4dDesc<T>(
                &bias_desc_, data_format(),
                vec64_t({ 1, num_output_, 1, 1 })
            );
        } else if (data_format() == "NHWC") {
            CuDNNSetTensor4dDesc<T>(
                &bias_desc_, data_format(),
                vec64_t({ 1, 1, 1, num_output_ })
            );
        }
    }

    auto* dy = X(-1).template data<T, Context>();

    if (Y(2)->name() != "NULL") {
        T* db = Y(2)->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            ctx()->cudnn_handle(),
            CuDNNType<T>::one,
            input_desc_, dy,
            CuDNNType<T>::zero,
            bias_desc_, db
        ));
    }

    if (Y(1)->name() != "NULL") {
        auto* x = X(0).template data<T, Context>();
        auto* dw = Y(1)->template mutable_data<T, Context>();
        kernel::DepthwiseConv2dWGrad(
            X(0).dim(0), channels_,
            in_shape_[0], in_shape_[1],
            out_shape_[0], out_shape_[1],
            kshape_[0], kshape_[1],
            stride_[0], stride_[1],
            pad_l_[0], pad_l_[1],
            dilation_[0], dilation_[1],
            data_format(),
            dy, x,
            dw, ctx()
        );
    }

    if (Y(0)->name() != "NULL") {
        auto* w = X(1).template data<T, Context>();
        auto* dx = Y(0)->template mutable_data<T, Context>();
        kernel::DepthwiseConv2dGrad(
            X(0).dim(0), channels_,
            in_shape_[0], in_shape_[1],
            out_shape_[0], out_shape_[1],
            kshape_[0], kshape_[1],
            stride_[0], stride_[1],
            pad_l_[0], pad_l_[1],
            dilation_[0], dilation_[1],
            data_format(),
            dy, w,
            dx, ctx()
        );
    }
}

template <class Context>
void CuDNNDepthwiseConv2dGradientOp<Context>::RunOnDevice() {
    group_ = channels_ = data_format()
        == "NCHW" ? X(0).dim(1) : X(0).dim(-1);
    ConvOpBase<Context>::Reshape(true);

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

DEPLOY_CUDNN(DepthwiseConv2d);
DEPLOY_CUDNN(DepthwiseConv2dGradient);

}  // namespace dragon

#endif  // WITH_CUDNN