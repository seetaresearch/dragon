#ifdef WITH_CUDNN

#include "operators/activation/sigmoid_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNSigmoidOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &Input(0));
    cudnnSetTensorDesc<T>(&output_desc, Output(0));
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnActivationForward(cudnn_handle(), act_desc,
                           CUDNNType<T>::one, input_desc, Xdata,
                       CUDNNType<T>::zero, output_desc, Ydata));
#else
    CUDNN_CHECK(cudnnActivationForward_v4(cudnn_handle(), act_desc,
                          CUDNNType<Dtype>::one, input_desc, Xdata,
                      CUDNNType<Dtype>::zero, output_desc, Ydata));
#endif
}

template <class Context>
void CuDNNSigmoidOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
#endif
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Sigmoid);

template <class Context> template <typename T>
void CuDNNSigmoidGradientOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &Input(-1));
    cudnnSetTensorDesc<T>(&output_desc, Output(0));
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnActivationBackward(cudnn_handle(), act_desc,
                            CUDNNType<T>::one, input_desc, Ydata,
                                              input_desc, dYdata,
                                              output_desc, Ydata,
                       CUDNNType<T>::zero, output_desc, dXdata));
#else
    CUDNN_CHECK(cudnnActivationBackward_v4(cudnn_handle(), act_desc,
                               CUDNNType<T>::one, input_desc, Ydata,
                                                 input_desc, dYdata,
                                                 output_desc, Ydata,
                          CUDNNType<T>::zero, output_desc, dXdata));
#endif
}

template <class Context>
void CuDNNSigmoidGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
#endif
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(SigmoidGradient);

}    // namespace dragon

#endif    // WITH_CUDNN