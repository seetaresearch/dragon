#ifdef WITH_CUDNN

#include "core/workspace.h"
#include "utils/filler.h"
#include "operators/vision/bias_add_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNBiasAddOp<Context>::RunImpl() {
    TENSOR_FILL(X(1), vec64_t({ axis_dim_ }));

    if (data_format() == "NCHW") {
        CuDNNSetTensor4dDesc<T>(
            &bias_desc_, data_format(),
            vec64_t({ 1, axis_dim_, 1, 1 })
        );
        CuDNNSetTensor4dDesc<T>(
            &output_desc_, data_format(),
            vec64_t({ outer_dim_, axis_dim_, 1, inner_dim_ })
        );
    } else if (data_format() == "NHWC") {
        CuDNNSetTensor4dDesc<T>(
            &bias_desc_, data_format(),
            vec64_t({ 1, 1, 1, axis_dim_ })
        );
        CuDNNSetTensor4dDesc<T>(
            &output_desc_, data_format(),
            vec64_t({ outer_dim_, 1, inner_dim_, axis_dim_ })
        );
    }

    auto* bias = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    // Copy X to Y firstly if necessary
    Y(0)->CopyFrom(X(0), ctx());

    CUDNN_CHECK(cudnnAddTensor(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        bias_desc_, bias,
        CuDNNType<T>::one,
        output_desc_, y
    ));
}

template <class Context>
void CuDNNBiasAddOp<Context>::RunOnDevice() {
    if (data_format() == "NCHW") {
        outer_dim_ = X(0).dim(0);
        axis_dim_ = X(0).dim(1);
        inner_dim_ = X(0).count(2);
    } else if (data_format() == "NHWC") {
        outer_dim_ = X(0).dim(0);
        axis_dim_ = X(0).dim(-1);
        inner_dim_ = X(0).count(1) / axis_dim_;
    } else {
        LOG(FATAL) << "Unknown DataFormat: "
                   << data_format();
    }

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

template <class Context> template <typename T>
void CuDNNBiasAddGradientOp<Context>::RunImpl() {
    if (data_format() == "NCHW") {
        CuDNNSetTensor4dDesc<T>(
            &input_desc_, data_format(),
            vec64_t({ outer_dim_, axis_dim_, 1, inner_dim_ })
        );
        CuDNNSetTensor4dDesc<T>(
            &bias_desc_, data_format(),
            vec64_t({ 1, axis_dim_, 1, 1 })
        );
    } else if (data_format() == "NHWC") {
        CuDNNSetTensor4dDesc<T>(
            &input_desc_, data_format(),
            vec64_t({ outer_dim_, 1, inner_dim_, axis_dim_ })
        );
        CuDNNSetTensor4dDesc<T>(
            &bias_desc_, data_format(),
            vec64_t({ 1, 1, 1, axis_dim_ })
        );
    }

    auto* dy = X(-1).template data<T, Context>();
    auto* db = Y(1)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnConvolutionBackwardBias(
        ctx()->cudnn_handle(),
        CuDNNType<T>::one,
        input_desc_, dy,
        CuDNNType<T>::zero,
        bias_desc_, db
    ));

    if (Y(0)->name() != "NULL" &&
        Y(0)->name() != X(-1).name()) {
        Y(0)->ReshapeLike(X(-1))
            ->CopyFrom(X(-1), ctx());
    }
}

template <class Context>
void CuDNNBiasAddGradientOp<Context>::RunOnDevice() {
    if (data_format() == "NCHW") {
        outer_dim_ = X(-1).dim(0);
        axis_dim_ = X(-1).dim(1);
        inner_dim_ = X(-1).count(2);
    } else if (data_format() == "NHWC") {
        outer_dim_ = X(-1).dim(0);
        axis_dim_ = X(-1).dim(-1);
        inner_dim_ = X(-1).count(1) / axis_dim_;
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }

    Y(1)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(-1));
}

DEPLOY_CUDNN(BiasAdd);
DEPLOY_CUDNN(BiasAddGradient);

}  // namespace dragon

#endif  // WITH_CUDNN