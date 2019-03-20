#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/bias_add_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNBiasAddOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), vector<int64_t>(1, dim));

    if (data_format == "NCHW") {
        cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
            vector<int64_t>({ 1, dim, 1, 1 }));
        cudnnSetTensor4dDesc<T>(&output_desc, data_format,
            vector<int64_t>({ outer_dim, dim, 1, inner_dim }));
    } else if (data_format == "NHWC") {
        cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
            vector<int64_t>({ 1, 1, 1, dim }));
        cudnnSetTensor4dDesc<T>(&output_desc, data_format,
            vector<int64_t>({ outer_dim, 1, inner_dim, dim }));
    }

    auto* Bdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    // Copy X to Y firstly if necessary
    Output(0)->template CopyFrom<Context>(Input(0), ctx());

    CUDNN_CHECK(cudnnAddTensor(ctx()->cudnn_handle(),
        CUDNNType<T>::one, bias_desc, Bdata,
            CUDNNType<T>::one, output_desc, Ydata));
}

template <class Context>
void CuDNNBiasAddOp<Context>::RunOnDevice() {
    if (data_format == "NCHW") {
        outer_dim = Input(0).dim(0);
        dim = Input(0).dim(1);
        inner_dim = Input(0).count(2);
    } else if (data_format == "NHWC") {
        outer_dim = Input(0).dim(0);
        dim = Input(0).dim(-1);
        inner_dim = Input(0).count(1) / dim;
    } else LOG(FATAL) << "Unknown data format: " << data_format;
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(BiasAdd);

template <class Context> template <typename T>
void CuDNNBiasAddGradientOp<Context>::RunWithType() {
    if (data_format == "NCHW") {
        cudnnSetTensor4dDesc<T>(&input_desc, data_format,
            vector<int64_t>({ outer_dim, dim, 1, inner_dim }));
        cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
            vector<int64_t>({ 1, dim, 1, 1 }));
    } else if (data_format == "NHWC") {
        cudnnSetTensor4dDesc<T>(&input_desc, data_format,
            vector<int64_t>({ outer_dim, 1, inner_dim, dim }));
        cudnnSetTensor4dDesc<T>(&bias_desc, data_format,
            vector<int64_t>({ 1, 1, 1, dim }));
    }

    auto* dYdata = Input(-1).template data<T, Context>();
    T* dBdata = Output(1)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnConvolutionBackwardBias(ctx()->cudnn_handle(),
        CUDNNType<T>::one, input_desc, dYdata,
            CUDNNType<T>::zero, bias_desc, dBdata));

    if (Output(0)->name() != "NULL" &&
        Output(0)->name() != Input(-1).name()) {
        Output(0)->ReshapeLike(Input(-1));
        Output(0)->template CopyFrom<Context>(Input(-1), ctx());
    }
}

template <class Context>
void CuDNNBiasAddGradientOp<Context>::RunOnDevice() {
    if (data_format == "NCHW") {
        outer_dim = Input(-1).dim(0);
        dim = Input(-1).dim(1);
        inner_dim = Input(-1).count(2);
    } else if (data_format == "NHWC") {
        outer_dim = Input(-1).dim(0);
        dim = Input(-1).dim(-1);
        inner_dim = Input(-1).count(1) / dim;
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    Output(1)->ReshapeLike(Input(0));

    if (XIsType(Input(-1), float)) RunWithType<float>();
    else if (XIsType(Input(-1), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(-1), { "float32", "float16" });
}

DEPLOY_CUDNN(BiasAddGradient);

}  // namespace dragon

#endif  // WITH_CUDNN