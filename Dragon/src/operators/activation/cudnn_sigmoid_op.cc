#ifdef WITH_CUDNN

#include "operators/activation/sigmoid_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNSigmoidOp<Context>::RunImpl() {
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
        CuDNNType<T>::one,
        input_desc_, x,
        CuDNNType<T>::zero,
        input_desc_, y
    ));
#endif
}

template <class Context>
void CuDNNSigmoidOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

template <class Context> template <typename T>
void CuDNNSigmoidGradientOp<Context>::RunImpl() {
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
void CuDNNSigmoidGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

DEPLOY_CUDNN(Sigmoid);
DEPLOY_CUDNN(SigmoidGradient);

}  // namespace dragon

#endif  // WITH_CUDNN