#ifdef WITH_CUDNN

#include "operators/vision/pooling_op.h"

namespace dragon {

template <class Context> template <typename T>
void CuDNNPooling2dOp<Context>::RunWithType() {
    cudnnSetTensor4dDesc<T>(&input_desc, this->data_format, &input(0));
    cudnnSetTensor4dDesc<T>(&output_desc, this->data_format, output(0));
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc,
                                            pool_mode,
                                  CUDNN_PROPAGATE_NAN,
           this->kernel_size[0], this->kernel_size[1],
                           this->pad[0], this->pad[1],
                   this->stride[0], this->stride[1]));
#else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc,
                                               pool_mode,
                                     CUDNN_PROPAGATE_NAN,
              this->kernel_size[0], this->kernel_size[1],
                              this->pad[0], this->pad[1],
                      this->stride[0], this->stride[1]));
#endif
    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();

    CUDNN_CHECK(cudnnPoolingForward(cudnn_handle(),
                                         pool_desc,
              CUDNNType<T>::one, input_desc, Xdata,
          CUDNNType<T>::zero, output_desc, Ydata));
}

template <class Context>
void CuDNNPooling2dOp<Context>::RunOnDevice() {
    Pooling2dOp<Context>::Reshape();

    if (input(0).template IsType<float>()) return RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) return RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CUDNN(Pooling2d);

template <class Context> template <typename T>
void CuDNNPooling2dGradientOp<Context>::RunWithType() {
    cudnnSetTensor4dDesc<T>(&input_desc, this->data_format, &input(-1));
    cudnnSetTensor4dDesc<T>(&output_desc, this->data_format, output(0));
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc,
                                            pool_mode,
                                  CUDNN_PROPAGATE_NAN,
           this->kernel_size[0], this->kernel_size[1],
                           this->pad[0], this->pad[1],
                   this->stride[0], this->stride[1]));
#else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(pool_desc,
                                               pool_mode,
                                     CUDNN_PROPAGATE_NAN,
              this->kernel_size[0], this->kernel_size[1],
                              this->pad[0], this->pad[1],
                      this->stride[0], this->stride[1]));
#endif
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
void CuDNNPooling2dGradientOp<Context>::RunOnDevice() {
    Pooling2dGradientOp<Context>::Reshape();

    if (input(0).template IsType<float>()) return RunWithType<float>();
#ifdef WITH_CUDA_FP16
    else if (input(0).template IsType<float16>()) return RunWithType<float16>();
#endif
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CUDNN(Pooling2dGradient);

}    // namespace dragon

#endif    // WITH_CUDNN