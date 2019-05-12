#ifdef WITH_CUDNN

#include "operators/vision/lrn_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNLRNOp<Context>::RunImpl() {
    if (data_format() == "NCHW") {
        CuDNNSetTensorDesc<T>(&input_desc_, &X(0));
        CuDNNSetTensorDesc<T>(&output_desc_, Y(0));

        auto* x = X(0).template data<T, Context>();
        auto* y = Y(0)->template mutable_data<T, Context>();

        CUDNN_CHECK(cudnnLRNCrossChannelForward(
            ctx()->cudnn_handle(), lrn_desc_,
            CUDNN_LRN_CROSS_CHANNEL_DIM1,
            CuDNNType<T>::one, input_desc_, x,
            CuDNNType<T>::zero, output_desc_, y
        ));
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
}

template <class Context>
void CuDNNLRNOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (this->mode_ == "ACROSS_CHANNELS") {
        if (XIsType(X(0), float)) {
            RunImpl<float>();
        } else if (XIsType(X(0), float16)) {
            RunImpl<float16>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float32", "float16" }
            );
        }
    } else if (this->mode_ == "WITHIN_CHANNEL") {
        LRNOp<Context>::RunOnDevice();
    } else {
        LOG(FATAL) << "Unknown Mode: " << this->mode_;
    }
}

template <class Context> template <typename T>
void CuDNNLRNGradientOp<Context>::RunImpl() {
    if (data_format() == "NCHW") {
        CuDNNSetTensorDesc<T>(&input_desc_, &X(-1));
        CuDNNSetTensorDesc<T>(&output_desc_, Y(0));

        auto* dy = X(-1).template data<T, Context>();
        auto* x = X(0).template data<T, Context>();
        auto* y = X(1).template data<T, Context>();
        auto* dx = Y(0)->template mutable_data<T, Context>();

        CUDNN_CHECK(cudnnLRNCrossChannelBackward(
            ctx()->cudnn_handle(), lrn_desc_,
            CUDNN_LRN_CROSS_CHANNEL_DIM1,
            CuDNNType<T>::one,
            input_desc_, y,
            input_desc_, dy,
            output_desc_, x,
            CuDNNType<T>::zero,
            output_desc_, dx
        ));
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
}

template <class Context>
void CuDNNLRNGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (this->mode_ == "ACROSS_CHANNELS") {
        if (XIsType(X(0), float)) {
            RunImpl<float>();
        } else if (XIsType(X(0), float16)) {
            RunImpl<float16>();
        } else {
            LOG(FATAL) << DTypeString(X(0),
                { "float32", "float16" }
            );
        }
    } else if (this->mode_ == "WITHIN_CHANNEL") {
        LRNGradientOp<Context>::RunOnDevice();
    } else {
        LOG(FATAL) << "Unknown Mode: " << this->mode_;
    }
}

DEPLOY_CUDNN(LRN);
DEPLOY_CUDNN(LRNGradient);

}  // namespace dragon

#endif  // WITH_CUDNN