#include "operators/activation/elu_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNEluOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(0));
    cudnnSetTensorDesc<T>(&output_desc, output(0));
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnActivationForward(cudnn_handle(), act_desc,
                           CUDNNType<T>::one, input_desc, Xdata,
                       CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNEluOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) return RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) return RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CUDNN(Elu);

template <class Context> template <typename T>
void CuDNNEluGradientOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(-1));
    cudnnSetTensorDesc<T>(&output_desc, output(0));
    auto* dYdata = input(-1).template data<T, Context>();
    auto* Ydata = input(0).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnActivationBackward(cudnn_handle(), act_desc,
                            CUDNNType<T>::one, input_desc, Ydata,
                                              input_desc, dYdata,
                                              output_desc, Ydata,
                       CUDNNType<T>::zero, output_desc, dXdata));

}

template <class Context>
void CuDNNEluGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));

    if (input(0).template IsType<float>()) return RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) return RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CUDNN(EluGradient);

}    // namespace dragon

#endif

#endif    // WITH_CUDNN