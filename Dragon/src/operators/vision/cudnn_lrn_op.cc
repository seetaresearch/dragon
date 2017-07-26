#ifdef WITH_CUDNN

#include "operators/vision/lrn_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNLRNOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(0));
    cudnnSetTensorDesc<T>(&output_desc, output(0));

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    CUDNN_CHECK(cudnnLRNCrossChannelForward(cudnn_handle(), 
                                                 norm_desc, 
                              CUDNN_LRN_CROSS_CHANNEL_DIM1,
                      CUDNNType<T>::one, input_desc, Xdata,
                  CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNLRNOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (this->mode == ACROSS_CHANNELS) {
        if (input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
        else if (input(0).template IsType<float16>()) RunWithType<float16>();
#endif
        else LOG(FATAL) << "unsupported input types."; 
    } else { 
        LRNOp<Context>::RunOnDevice(); 
    }
}

DEPLOY_CUDNN(LRN);

template <class Context> template <typename T>
void CuDNNLRNGradientOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(-1));
    cudnnSetTensorDesc<T>(&output_desc, output(0));

    auto* dYdata = input(-1).template data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = input(1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    CUDNN_CHECK(cudnnLRNCrossChannelBackward(cudnn_handle(), 
                                                  norm_desc, 
                               CUDNN_LRN_CROSS_CHANNEL_DIM1,
                       CUDNNType<T>::one, input_desc, Ydata, 
                                         input_desc, dYdata,
                                         output_desc, Xdata, 
                  CUDNNType<T>::zero, output_desc, dXdata));
}

template <class Context>
void CuDNNLRNGradientOp<Context>::RunOnDevice(){
    output(0)->ReshapeLike(input(0));

    if (this->mode == ACROSS_CHANNELS) {
        if (input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
        else if (input(0).template IsType<float16>()) RunWithType<float16>();
#endif
        else LOG(FATAL) << "unsupported input types."; 
    } else { 
        LRNGradientOp<Context>::RunOnDevice(); 
    }
}

DEPLOY_CUDNN(LRNGradient);

}    // namespace dragon

#endif    // WITH_CUDNN