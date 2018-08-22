#include "operators/activation/elu_op.h"

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

namespace dragon {

template <class Context> template <typename T>
void CuDNNEluOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &Input(0));
    cudnnSetTensorDesc<T>(&output_desc, Output(0));
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnActivationForward(
        ctx()->cudnn_handle(), act_desc,
            CUDNNType<T>::one, input_desc, Xdata,
                CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNEluOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
#endif
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(Elu);

template <class Context> template <typename T>
void CuDNNEluGradientOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &Input(-1));
    cudnnSetTensorDesc<T>(&output_desc, Output(0));
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* Ydata = Input(0).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnActivationBackward(
        ctx()->cudnn_handle(), act_desc,
            CUDNNType<T>::one, input_desc, Ydata,
                input_desc, dYdata, output_desc, Ydata,
                    CUDNNType<T>::zero, output_desc, dXdata));
}

template <class Context>
void CuDNNEluGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
#endif
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CUDNN(EluGradient);

}    // namespace dragon

#endif

#endif    // WITH_CUDNN