#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/conv_op.h"

namespace dragon {

#define WORKSPACE_LIMIT_BYTES 64 * 1024 * 1024 // 64MB

template <class Context> template <typename T>
void CuDNNConv2dOp<Context>::ResetDesc() {
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc, CUDNNType<T>::type, format,
            num_output / cudnn_group, channels / group,
                kernel_shape[0], kernel_shape[1]));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
        filter_desc, CUDNNType<T>::type, format,
            num_output / cudnn_group, channels / group,
                kernel_shape[0], kernel_shape[1]));
#endif

    // Determine the input & output shape
    input_dims = Input(0).dims();
    cudnnSetTensor4dDescWithGroup<T>(
        &input_desc, data_format, Input(0).dims(), cudnn_group);
    cudnnSetTensor4dDescWithGroup<T>(
        &output_desc, data_format, Output(0)->dims(), cudnn_group);

    // Determine the bias shape
    if (HasBias()) {
        cudnnSetTensor4dDesc<T>(
            &output2b_desc, data_format, Output(0)->dims());
        if (data_format == "NCHW") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, num_output, 1, 1 }));
        } else if (data_format == "NHWC") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, 1, 1, num_output }));
        }
    }

    // Determine the misc
    if (data_format == "NCHW") {
        x_offset = Input(0).count(1) / cudnn_group;
        y_offset = Output(0)->count(1) / cudnn_group;
    } else if (data_format == "NHWC") {
        x_offset = Input(0).dim(-1) / cudnn_group;
        y_offset = Output(0)->dim(-1) / cudnn_group;
    }

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        ctx()->cudnn_handle(), input_desc,
            filter_desc, conv_desc, output_desc,
                CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                    WORKSPACE_LIMIT_BYTES, &fwd_algo));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        ctx()->cudnn_handle(), input_desc,
            filter_desc, conv_desc, output_desc,
                fwd_algo, &fwd_data_size));
}

template <class Context> template <typename T>
void CuDNNConv2dOp<Context>::RunWithType() {
    if (Input(0).dims() != input_dims) ResetDesc<T>();

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    TENSOR_FILL(Input(1), weight_shape);
    auto* Wdata = Input(1).template data<T, Context>();
    if (HasBias()) TENSOR_FILL(Input(2), bias_shape);

    auto* WSdata = (uint8_t*)ws()->template
        caches<Context>({ fwd_data_size })[0];

    auto cudnn_handle = ctx()->cudnn_handle();

    for (int g = 0; g < cudnn_group; g++) {
        CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle,
            CUDNNType<T>::one, input_desc, Xdata + x_offset * g,
                filter_desc, Wdata + weight_offset * g,
                    conv_desc, fwd_algo, WSdata, fwd_data_size,
            CUDNNType<T>::zero, output_desc, Ydata + y_offset * g));
    }

    if (HasBias()) {
        auto* Bdata = Input(2).template data<T, Context>();
        CUDNN_CHECK(cudnnAddTensor(cudnn_handle,
            CUDNNType<T>::one, bias_desc, Bdata,
                CUDNNType<T>::one, output2b_desc, Ydata));
    }
}

template <class Context>
void CuDNNConv2dOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < dilation.size(); i++)
        if (dilation[i] != 1)
            return Conv2dOp<Context>::RunOnDevice();
#endif
    Conv2dOp<Context>::Reshape();

    if (XIsType(Input(0), float)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1],
                    dilation[0], dilation[1],
                        CUDNN_CROSS_CORRELATION,
                            CUDNN_DATA_FLOAT));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1], 1, 1,
                    CUDNN_CROSS_CORRELATION));
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group));
        if (enable_tensor_core)
            CUDNN_CHECK(cudnnSetConvolutionMathType(
                conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
        RunWithType<float>();
    } else if (XIsType(Input(0), float16)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        compute_type = CUDNN_DATA_FLOAT;
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1],
                    dilation[0], dilation[1],
                        CUDNN_CROSS_CORRELATION,
                            compute_type));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1], 1, 1,
                    CUDNN_CROSS_CORRELATION));
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group));
        if (enable_tensor_core)
            CUDNN_CHECK(cudnnSetConvolutionMathType(
                conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
        RunWithType<float16>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Conv2d);

template <class Context> template <typename T>
void CuDNNConv2dGradientOp<Context>::ResetDesc() {
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc, CUDNNType<T>::type, format,
            num_output / cudnn_group, channels / group,
                kernel_shape[0], kernel_shape[1]));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
        filter_desc, CUDNNType<T>::type, format,
            num_output / cudnn_group, channels / group,
                kernel_shape[0], kernel_shape[1]));
#endif

    // Determine the input & output shape
    input_dims = Input(0).dims();
    cudnnSetTensor4dDescWithGroup<T>(
        &input_desc, data_format, Input(-1).dims(), cudnn_group);
    cudnnSetTensor4dDescWithGroup<T>(
        &output_desc, data_format, Input(0).dims(), cudnn_group);

    // Determine the bias shape
    if (HasBias()) {
        cudnnSetTensor4dDesc<T>(
            &input2b_desc, data_format, Input(-1).dims());
        if (data_format == "NCHW") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, num_output, 1, 1 }));
        } else if (data_format == "NHWC") {
            cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
                vector<int64_t>({ 1, 1, 1, num_output }));
        }
    }

    // Determine the misc
    if (data_format == "NCHW") {
        x_offset = Input(0).count(1) / cudnn_group;
        y_offset = Input(-1).count(1) / cudnn_group;
    } else if (data_format == "NHWC") {
        x_offset = Input(0).dim(-1) / cudnn_group;
        y_offset = Input(-1).dim(-1) / cudnn_group;
    }

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
        ctx()->cudnn_handle(), output_desc,
            input_desc, conv_desc, filter_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                    WORKSPACE_LIMIT_BYTES, &bwd_filter_algo));

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        ctx()->cudnn_handle(), output_desc,
            input_desc, conv_desc, filter_desc,
                bwd_filter_algo, &bwd_filter_size));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        ctx()->cudnn_handle(), filter_desc,
            input_desc, conv_desc, output_desc,
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                    WORKSPACE_LIMIT_BYTES, &bwd_data_algo));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        ctx()->cudnn_handle(), filter_desc,
            input_desc, conv_desc, output_desc,
                bwd_data_algo, &bwd_data_size));
}

template <class Context> template <typename T>
void CuDNNConv2dGradientOp<Context>::RunWithType() {
    if (Input(0).dims() != input_dims) ResetDesc<T>();

    const T* dYdata = Input(2).template data<T, Context>();

    auto* WSdata = ws()->template caches<Context>({
        std::max(bwd_data_size, bwd_filter_size)})[0];

    auto cudnn_handle = ctx()->cudnn_handle();

    if (Output(2)->name() != "ignore") {
        T* dBdata = Output(2)->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_handle,
            CUDNNType<T>::one, input2b_desc, dYdata,
                CUDNNType<T>::one, bias_desc, dBdata));
    }

    for (int g = 0; g < cudnn_group; g++) {
        if (Output(1)->name() != "ignore") {
            auto* Xdata = Input(0).template data<T, Context>();
            auto* dWdata = Output(1)->template mutable_data<T, Context>();
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn_handle,
                CUDNNType<T>::one, output_desc, Xdata + x_offset * g,
                    input_desc, dYdata + y_offset * g,
                        conv_desc, bwd_filter_algo, WSdata, bwd_filter_size,
                CUDNNType<T>::one, filter_desc, dWdata + weight_offset * g));
        }
        if (Output(0)->name() != "ignore") {
            auto* Wdata = Input(1).template data<T, Context>();
            auto* dXdata = Output(0)->template mutable_data<T, Context>();
            CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn_handle,
                CUDNNType<T>::one, filter_desc, Wdata + weight_offset * g,
                    input_desc, dYdata + y_offset * g,
                        conv_desc, bwd_data_algo, WSdata, bwd_data_size,
                CUDNNType<T>::zero, output_desc, dXdata + x_offset * g));
        }
    }
}

template <class Context>
void CuDNNConv2dGradientOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < dilation.size(); i++)
        if (dilation[i] != 1)
            return Conv2dGradientOp<Context>::RunOnDevice();
#endif
    Conv2dGradientOp<Context>::GradientReshape();

    if (XIsType(Input(0), float)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1],
                    dilation[0], dilation[1],
                        CUDNN_CROSS_CORRELATION,
                            CUDNN_DATA_FLOAT));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1], 1, 1,
                    CUDNN_CROSS_CORRELATION));
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group));
        if (enable_tensor_core)
            CUDNN_CHECK(cudnnSetConvolutionMathType(
                conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
        RunWithType<float>();
    } else if (XIsType(Input(0), float16)) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        compute_type = CUDNN_DATA_FLOAT;
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1],
                    dilation[0], dilation[1],
                        CUDNN_CROSS_CORRELATION,
                            compute_type));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            pad_l[0], pad_l[1],
                stride[0], stride[1], 1, 1,
                    CUDNN_CROSS_CORRELATION));
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group));
        if (enable_tensor_core)
            CUDNN_CHECK(cudnnSetConvolutionMathType(
                conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
        RunWithType<float16>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Conv2dGradient);

}  // namespace dragon

#endif  // WITH_CUDNN