#ifdef WITH_CUDNN

#include "operators/activation/relu_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNReluOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(&input_desc_, &X(0));
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnActivationForward(
        ctx()->cudnn_handle(),
        act_desc_,
        CuDNNType<T>::one,
        input_desc_, x,
        CuDNNType<T>::zero,
        input_desc_, y
    ));
#else
    CUDNN_CHECK(cudnnActivationForward_v4(
        ctx()->cudnn_handle(),
        act_desc_,
        CuDNNType<Dtype>::one,
        input_desc_, x,
        CuDNNType<Dtype>::zero,
        input_desc_, y
    ));
#endif
}

template <class Context>
void CuDNNReluOp<Context>::RunOnDevice() {
    if (this->slope_ != 0) {
        // CuDNN does not support LeakyRelu
        return ReluOp<Context>::RunOnDevice();
    }

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context> template <typename T>
void CuDNNReluGradientOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(&input_desc_, &X(1));
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnActivationBackward(
        ctx()->cudnn_handle(),
        act_desc_,
        CuDNNType<T>::one,
        input_desc_, y,
        input_desc_, dy,
        input_desc_, y,
        CuDNNType<T>::zero,
        input_desc_, dx
    ));
#else
    CUDNN_CHECK(cudnnActivationBackward_v4(
        ctx()->cudnn_handle(),
        act_desc_,
        CuDNNType<T>::one,
        input_desc_, y,
        input_desc_, dy,
        input_desc_, y,
        CuDNNType<T>::zero,
        input_desc_, dx
    ));
#endif
}

template <class Context>
void CuDNNReluGradientOp<Context>::RunOnDevice() {
    if (this->slope_ != 0) {
        // CuDNN does not support LeakyRelu
        return ReluGradientOp<Context>::RunOnDevice();
    }

    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CUDNN(Relu);
DEPLOY_CUDNN(ReluGradient);

}  // namespace dragon

#endif  // WITH_CUDNN