#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/depthwise_conv_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNDepthwiseConv2dOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), weight_shape);
    if (HasBias()) {
        TENSOR_FILL(Input(2), bias_shape);
        if (data_format == "NCHW") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, num_output, 1, 1 }));
        } else if (data_format == "NHWC") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, 1, 1, num_output }));
        }
    }

    cudnnSetTensor4dDesc<T>(&output_desc,
        data_format, Output(0)->dims());

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Wdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::DepthwiseConv2d(
        Input(0).dim(0), channels,
        input_shape[0], input_shape[1],
        output_shape[0], output_shape[1],
        kernel_shape[0], kernel_shape[1],
        stride[0], pad_l[0], pad_l[1],
        data_format, Xdata, Wdata, Ydata, ctx());

    if (HasBias()) {
        auto* Bdata = Input(2).template data<T, Context>();
        CUDNN_CHECK(cudnnAddTensor(
            ctx()->cudnn_handle(),
            CUDNNType<T>::one, bias_desc, Bdata,
            CUDNNType<T>::one, output_desc, Ydata));
    }
}

template <class Context>
void CuDNNDepthwiseConv2dOp<Context>::RunOnDevice() {
    group = channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
    CHECK_EQ(channels, num_output)
        << "Excepted in/out channels unchanged.";
    Reshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CUDNN(DepthwiseConv2d);

template <class Context> template <typename T>
void CuDNNDepthwiseConv2dGradientOp<Context>::RunWithType() {
    if (HasBias()) {
        cudnnSetTensor4dDesc<T>(
            &input_desc, data_format, Input(-1).dims());
        if (data_format == "NCHW") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, num_output, 1, 1 }));
        } else if (data_format == "NHWC") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, 1, 1, num_output }));
        }
    }

    auto* dYdata = Input(-1).template data<T, Context>();

    if (HasBias()) {
        T* dBdata = Output(2)->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnConvolutionBackwardBias(ctx()->cudnn_handle(),
            CUDNNType<T>::one, input_desc, dYdata,
                CUDNNType<T>::zero, bias_desc, dBdata));
    }

    for (int n = 0; n < Input(2).dim(0); n++) {
        if (Output(1)->name() != "NULL") {
            auto* Xdata = Input(0).template data<T, Context>();
            auto* dWdata = Output(1)->template mutable_data<T, Context>();
            math::Set(Output(1)->count(), cast::to<T>(0.f), dWdata, ctx());
            kernel::DepthwiseConv2dWGrad(
                Input(0).dim(0), channels,
                input_shape[0], input_shape[1],
                output_shape[0], output_shape[1],
                kernel_shape[0], kernel_shape[1],
                stride[0], pad_l[0], pad_l[1],
                data_format, dYdata, Xdata, dWdata, ctx());
        }
        if (Output(0)->name() != "NULL") {
            auto* Wdata = Input(1).template data<T, Context>();
            auto* dXdata = Output(0)->template mutable_data<T, Context>();
            kernel::DepthwiseConv2dGrad(
                Input(0).dim(0), channels,
                input_shape[0], input_shape[1],
                output_shape[0], output_shape[1],
                kernel_shape[0], kernel_shape[1],
                stride[0], pad_l[0], pad_l[1],
                data_format, dYdata, Wdata, dXdata, ctx());
        }
    }
}

template <class Context>
void CuDNNDepthwiseConv2dGradientOp<Context>::RunOnDevice() {
    group = channels = data_format == "NCHW" ?
        Input(0).dim(1) : Input(0).dim(-1);
#if CUDNN_VERSION_MIN(7, 0, 0)
    // The group implementation of CuDNN is faster
    // Enable if CuDNN >= 7.0
    return CuDNNConv2dGradientOp<Context>::RunOnDevice();
#endif
    GradientReshape();

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CUDNN(DepthwiseConv2dGradient);

}  // namespace dragon

#endif  // WITH_CUDNN