#ifdef WITH_CUDNN

#include "operators/vision/pool_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNPool2dOp<Context>::RunImpl() {
    CuDNNSetTensor4dDesc<T>(
        &input_desc_,
        data_format(),
        &X(0)
    );
    CuDNNSetTensor4dDesc<T>(
        &output_desc_,
        data_format(),
        Y(0)
    );
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        pool_mode_,
        CUDNN_PROPAGATE_NAN,
        this->kshape_[0], this->kshape_[1],
        this->pad_l_[0], this->pad_l_[1],
        this->stride_[0], this->stride_[1]
    ));
#else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(
        pool_desc_,
        pool_mode_,
        CUDNN_PROPAGATE_NAN,
        this->kshape_[0], this->kshape_[1],
        this->pad_l_[0], this->pad_l_[1],
        this->stride_[0], this->stride_[1]
    ));
#endif
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnPoolingForward(
        ctx()->cudnn_handle(),
        pool_desc_,
        CuDNNType<T>::one,
        input_desc_, x,
        CuDNNType<T>::zero,
        output_desc_, y
    ));
}

template <class Context>
void CuDNNPool2dOp<Context>::RunOnDevice() {
    Pool2dOp<Context>::Reshape();

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

template <class Context> template <typename T>
void CuDNNPool2dGradientOp<Context>::RunImpl() {
    CuDNNSetTensor4dDesc<T>(
        &input_desc_,
        data_format(),
        &X(-1)
    );
    CuDNNSetTensor4dDesc<T>(
        &output_desc_,
        data_format(),
        Y(0)
    );
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        pool_mode_,
        CUDNN_PROPAGATE_NAN,
        this->kshape_[0], this->kshape_[1],
        this->pad_l_[0], this->pad_l_[1],
        this->stride_[0], this->stride_[1]
    ));
#else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(
        pool_desc_,
        pool_mode_,
        CUDNN_PROPAGATE_NAN,
        this->kshape_[0], this->kshape_[1],
        this->pad_l_[0], this->pad_l_[1],
        this->stride_[0], this->stride_[1]
    ));
#endif
    auto* x = X(0).template data<T, Context>();
    auto* y = X(1).template data<T, Context>();
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnPoolingBackward(
        ctx()->cudnn_handle(),
        pool_desc_,
        CuDNNType<T>::one,
        input_desc_, y,
        input_desc_, dy,
        output_desc_, x,
        CuDNNType<T>::zero,
        output_desc_, dx
    ));
}

template <class Context>
void CuDNNPool2dGradientOp<Context>::RunOnDevice() {
    Pool2dGradientOp<Context>::Reshape();

    DispatchHelper<TensorTypes
        <float, float16>>::Call(this, X(0));
}

DEPLOY_CUDNN(Pool2d);
DEPLOY_CUDNN(Pool2dGradient);

}  // namespace dragon

#endif  // WITH_CUDNN