#ifdef WITH_CUDNN

#include "operators/activation/softmax_op.h"

namespace dragon {

#define DETERMINE_RUNTIME_ARGS(X) \
    axis_ = OpArg<int64_t>("axis", 0); \
    axis_ = axis_ < 0 ? axis_ + X.ndim() : axis_; \
    CHECK(axis_ >= 0 && axis_ < X.ndim()) \
       << "\nExcepted the axis in [-" << X.ndim() \
       << ", " << X.ndim() << "), got " \
       << OpArg<int64_t>("axis", 0) << ".";

template <class Context> template <typename T>
void CuDNNSoftmaxOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(
        &input_desc_,
        vec64_t({
            outer_dim_,
            X(0).dim(axis_),
            inner_dim_
        })
    );

    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnSoftmaxForward(
        ctx()->cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        CuDNNType<T>::one,
        input_desc_, x,
        CuDNNType<T>::zero,
        input_desc_, y
    ));
}

template <class Context>
void CuDNNSoftmaxOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

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
void CuDNNSoftmaxGradientOp<Context>::RunImpl() {
    CuDNNSetTensorDesc<T>(
        &input_desc_,
        vec64_t({
            outer_dim_,
            X(0).dim(axis_),
            inner_dim_
        })
    );

    auto* y  = X(0).template data<T, Context>();
    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnSoftmaxBackward(
        ctx()->cudnn_handle(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        CuDNNType<T>::one,
        input_desc_, y,
        input_desc_, dy,
        CuDNNType<T>::zero,
        input_desc_, dx
    ));
}

template <class Context>
void CuDNNSoftmaxGradientOp<Context>::RunOnDevice() {
    DETERMINE_RUNTIME_ARGS(X(0));
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

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

DEPLOY_CUDNN(Softmax);
DEPLOY_CUDNN(SoftmaxGradient);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon

#endif  // WITH_CUDNN