#ifdef WITH_CUDNN

#include "operators/vision/lrn_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNLRNOp<Context>::RunWithType() {
    if (this->data_format == "NCHW") {
        cudnnSetTensorDesc<T>(&input_desc, &Input(0));
        cudnnSetTensorDesc<T>(&output_desc, Output(0));
        auto* Xdata = Input(0).template data<T, Context>();
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnLRNCrossChannelForward(cudnn_handle(),
                                                     norm_desc,
                                  CUDNN_LRN_CROSS_CHANNEL_DIM1,
                          CUDNNType<T>::one, input_desc, Xdata,
                      CUDNNType<T>::zero, output_desc, Ydata));
    } else LOG(FATAL) << "Unknown data format: " << this->data_format;
}

template <class Context>
void CuDNNLRNOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (this->mode == "ACROSS_CHANNELS") {
#ifdef WITH_CUDA_FP16
        if (XIsType(Input(0), float)) RunWithType<float>();
        else if (XIsType(Input(0), float16)) RunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
#else
        if (XIsType(Input(0), float)) RunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
#endif
    } else if (this->mode == "WITHIN_CHANNEL") {
        LRNOp<Context>::RunOnDevice(); 
    } else {
        LOG(FATAL) << "Unsupported lrn mode: " << this->mode;
    }
}

DEPLOY_CUDNN(LRN);

template <class Context> template <typename T>
void CuDNNLRNGradientOp<Context>::RunWithType() {
    if (this->data_format == "NCHW") {
        cudnnSetTensorDesc<T>(&input_desc, &Input(-1));
        cudnnSetTensorDesc<T>(&output_desc, Output(0));

        auto* dYdata = Input(-1).template data<T, Context>();
        auto* Xdata = Input(0).template data<T, Context>();
        auto* Ydata = Input(1).template data<T, Context>();
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        CUDNN_CHECK(cudnnLRNCrossChannelBackward(cudnn_handle(),
                                                      norm_desc,
                                   CUDNN_LRN_CROSS_CHANNEL_DIM1,
                           CUDNNType<T>::one, input_desc, Ydata,
                                             input_desc, dYdata,
                                             output_desc, Xdata,
                      CUDNNType<T>::zero, output_desc, dXdata));
    } else LOG(FATAL) << "Unknown data format: " << this->data_format;
}

template <class Context>
void CuDNNLRNGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (this->mode == "ACROSS_CHANNELS") {
#ifdef WITH_CUDA_FP16
        if (XIsType(Input(0), float)) RunWithType<float>();
        else if (XIsType(Input(0), float16)) RunWithType<float16>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
#else
        if (XIsType(Input(0), float)) RunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
#endif
    } else if (this->mode == "WITHIN_CHANNEL") {
        LRNGradientOp<Context>::RunOnDevice(); 
    } else {
        LOG(FATAL) << "Unsupported lrn mode: " << this->mode;
    }
}

DEPLOY_CUDNN(LRNGradient);

}    // namespace dragon

#endif    // WITH_CUDNN