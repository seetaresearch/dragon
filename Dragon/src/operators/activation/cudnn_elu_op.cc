#include "operators/activation/elu_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNEluOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(&input_desc_, &X(0));
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnActivationForward(
        ctx()->cudnn_handle(),
        act_desc_,
        CuDNNType<T>::one,
        input_desc_, x,
        CuDNNType<T>::zero,
        input_desc_, y
    ));
}

template <class Context>
void CuDNNEluOp<Context>::RunOnDevice() {
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
void CuDNNEluGradientOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(&input_desc_, &X(1));
    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

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
}

template <class Context>
void CuDNNEluGradientOp<Context>::RunOnDevice() {
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

DEPLOY_CUDNN(Elu);
DEPLOY_CUDNN(EluGradient);

}  // namespace dragon

#endif  // CUDNN_VERSION_MIN(6, 0, 0)

#endif  // WITH_CUDNN