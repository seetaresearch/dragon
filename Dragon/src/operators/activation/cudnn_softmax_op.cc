#ifdef WITH_CUDNN

#include "operators/activation/softmax_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNSoftmaxOp<Context>::RunWithType() {
    Tensor fake_tensor;
    fake_tensor.Reshape(vector<TIndex>({ outer_dim, input(0).dim(axis), inner_dim }));
    cudnnSetTensorDesc<T>(&input_desc, &fake_tensor);
    cudnnSetTensorDesc<T>(&output_desc, &fake_tensor);

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL, CUDNNType<T>::one, input_desc, Xdata,
        CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNSoftmaxOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (int)input(0).ndim() - 1;
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));
    
    if (input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CUDNN(Softmax);

template <class Context> template <typename T>
void CuDNNSoftmaxGradientOp<Context>::RunWithType() {
    Tensor fake_tensor;
    fake_tensor.Reshape(vector<TIndex>({ outer_dim, input(0).dim(axis), inner_dim }));
    cudnnSetTensorDesc<T>(&input_desc, &fake_tensor);
    cudnnSetTensorDesc<T>(&output_desc, &fake_tensor);

    auto* dYdata = input(-1).template data<T, Context>();
    auto* Ydata = input(0).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    CUDNN_CHECK(cudnnSoftmaxBackward(cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL, CUDNNType<T>::one, input_desc, Ydata,
        input_desc, dYdata, CUDNNType<T>::zero, output_desc, dXdata));
}

template <class Context>
void CuDNNSoftmaxGradientOp<Context>::RunOnDevice() {
    if (axis == -1) axis = (int)input(0).ndim() - 1;
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    output(0)->ReshapeLike(input(0));
    

    if (input(0).template IsType<float>()) RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
#endif
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CUDNN(SoftmaxGradient);

}    // namespace dragon

#endif    // WITH_CUDNN


