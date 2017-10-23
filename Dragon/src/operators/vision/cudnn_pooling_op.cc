#ifdef WITH_CUDNN

#include "operators/vision/pooling_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNPoolingOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(0));
    cudnnSetTensorDesc<T>(&output_desc, output(0));
    if (this->global_pooling) {
#if CUDNN_VERSION_MIN(5, 0, 0)
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc,
                                                pool_mode,
                                      CUDNN_PROPAGATE_NAN,
                         input(0).dim(2), input(0).dim(3),
                                                     0, 0,
                                                   1, 1));
#else
        CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc, 
                                                   pool_mode,
                                         CUDNN_PROPAGATE_NAN, 
                            input(0).dim(2), input(0).dim(3),
                                                        0, 0,
                                                      1, 1));
#endif
    }
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnPoolingForward(cudnn_handle(), 
                                         pool_desc,
              CUDNNType<T>::one, input_desc, Xdata,
          CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNPoolingOp<Context>::RunOnDevice() {
    PoolingOp<Context>::Reshape();

    if (input(0).template IsType<float>()) return RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) return RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CUDNN(Pooling);

template <class Context> template <typename T>
void CuDNNPoolingGradientOp<Context>::RunWithType() {
    cudnnSetTensorDesc<T>(&input_desc, &input(-1));
    cudnnSetTensorDesc<T>(&output_desc, output(0));
    if (this->global_pooling) {
#if CUDNN_VERSION_MIN(5, 0, 0)
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc,
                                                pool_mode,
                                      CUDNN_PROPAGATE_NAN,
                         input(0).dim(2), input(0).dim(3),
                                                     0, 0,
                                                   1, 1));
#else
        CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc, 
                                                   pool_mode,
                                         CUDNN_PROPAGATE_NAN, 
                            input(0).dim(2), input(0).dim(3),
                                                        0, 0,
                                                      1, 1));
#endif
    }
    auto* dYdata = input(-1).template data<T, Context>();
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = input(1).template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnPoolingBackward(cudnn_handle(), 
                                          pool_desc,
               CUDNNType<T>::one, input_desc, Ydata, 
                                 input_desc, dYdata,
                                 output_desc, Xdata, 
          CUDNNType<T>::zero, output_desc, dXdata));
}

template <class Context>
void CuDNNPoolingGradientOp<Context>::RunOnDevice() {
    PoolingGradientOp<Context>::Reshape();

    if (input(0).template IsType<float>()) return RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) return RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CUDNN(PoolingGradient);

}    // namespace dragon

#endif    // WITH_CUDNN