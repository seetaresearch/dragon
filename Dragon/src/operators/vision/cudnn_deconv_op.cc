#ifdef WITH_CUDNN

#include "operators/vision/deconv_op.h"
#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"

namespace dragon {

#define WORKSPACE_LIMIT_BYTES 64 * 1024 * 1024 // 64MB

template <class Context> template <typename T>
void CuDNNDeConvOp<Context>::RunWithType() {
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, 
                                    CUDNNType<T>::type, 
                                     CUDNN_TENSOR_NCHW,
                          this->channels / this->group,
                        this->num_output / this->group,
          this->kernel_size[0], this->kernel_size[1]));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc, 
                                       CUDNNType<T>::type, 
                                        CUDNN_TENSOR_NCHW,
                             this->channels / this->group, 
                           this->num_output / this->group,
             this->kernel_size[0], this->kernel_size[1]));
#endif
    cudnnSetTensorDesc<T>(&input_desc,
                          vector<TIndex>({ input(0).dim(0), 
                                           input(0).dim(1) / this->group,
                                           input(0).dim(2), 
                                           input(0).dim(3) }),
                          vector<TIndex>({ input(0).count(1), 
                                           input(0).count(2),
                                           input(0).count(3), 
                                           1 }));
    cudnnSetTensorDesc<T>(&output_desc,
                          vector<TIndex>({ output(0)->dim(0), 
                                           output(0)->dim(1) / this->group,
                                           output(0)->dim(2), 
                                           output(0)->dim(3) }),
                          vector<TIndex>({ output(0)->count(1), 
                                           output(0)->count(2),
                                           output(0)->count(3), 
                                           1 }));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle[0],
                                                       filter_desc, 
                                                        input_desc, 
                                                         conv_desc, 
                                                       output_desc,
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                             WORKSPACE_LIMIT_BYTES, 
                                                       &fwd_algo));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle[0],
                                                           filter_desc, 
                                                            input_desc, 
                                                             conv_desc, 
                                                           output_desc,
                                                              fwd_algo, 
                                            &workspace_fwd_data_size));

    Tensor* buffer = ws()->GetBuffer();
    if (workspace_fwd_data_size == 0) workspace_fwd_data_size += 1;
    buffer->Reshape(vector<TIndex>(1, this->group * workspace_fwd_data_size));
    if (InputSize() > 2) {
        bias_offset = this->num_output / this->group;
        cudnnSetTensorDesc<T>(&bias_desc, vector<TIndex>({ 1, bias_offset, 1, 1 }));
    }

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    TENSOR_FILL(input(1), this->weight_shape);
    auto* Wdata = input(1).template data<T, Context>();
    if (InputSize() > 2) TENSOR_FILL(input(2), this->bias_shape);

    for (int g = 0; g < this->group; g++) {
        auto* workspace = buffer->template mutable_data<char, Context>();
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle[g],
                            CUDNNType<T>::one, filter_desc, Wdata + this->weight_offset * g,
                                                     input_desc, Xdata + this->x_offset * g, 
                                                                                  conv_desc, 
                                                                                   fwd_algo,
                           workspace + g * workspace_fwd_data_size, workspace_fwd_data_size,
                              CUDNNType<T>::zero, output_desc, Ydata + this->y_offset * g));

        if (InputSize() > 2) {
            auto* bias = input(2).template data<T, Context>();
            CUDNN_CHECK(cudnnAddTensor(handle[g],
                CUDNNType<T>::one, bias_desc, bias + this->bias_offset * g,
                CUDNNType<T>::one, output_desc, Ydata + this->y_offset * g));
        }
    }
    kernel::Empty<T, Context>();
    ws()->ReleaseBuffer(buffer);
}

template <class Context>
void CuDNNDeConvOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < this->dilation.size(); i++)
        if (this->dilation[i] != 1) return DeConvOp<Context>::RunOnDevice();
#endif
    DeConvOp<Context>::Reshape();
    this->x_offset /= this->group;
    this->y_offset /= this->group;

    if (input(0).template IsType<float>()) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                         this->dilation[0], this->dilation[1],
                                      CUDNN_CROSS_CORRELATION, 
                                           CUDNN_DATA_FLOAT));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                                                         1, 1, 
                                    CUDNN_CROSS_CORRELATION));
#endif
        RunWithType<float>();
    } else if (input(0).template IsType<float16>()) {
#ifdef WITH_CUDA_FP16
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                         this->dilation[0], this->dilation[1], 
                                      CUDNN_CROSS_CORRELATION, 
                                           CUDNN_DATA_FLOAT));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                                                         1, 1, 
                                    CUDNN_CROSS_CORRELATION));
#endif
        RunWithType<float16>();
#endif  // WITH_CUDA_FP16
    } else { LOG(FATAL) << "Unsupported input types."; }
}

DEPLOY_CUDNN(DeConv);

template <class Context> template <typename T>
void CuDNNDeConvGradientOp<Context>::RunWithType() {
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, 
                                    CUDNNType<T>::type, 
                                     CUDNN_TENSOR_NCHW,
                          this->channels / this->group, 
                        this->num_output / this->group,
          this->kernel_size[0], this->kernel_size[1]));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc, 
                                       CUDNNType<T>::type, 
                                        CUDNN_TENSOR_NCHW,
                             this->channels / this->group, 
                           this->num_output / this->group,
             this->kernel_size[0], this->kernel_size[1]));
#endif
    cudnnSetTensorDesc<T>(&input_desc,
                          vector<TIndex>({ input(-1).dim(0), 
                                           input(-1).dim(1) / this->group,
                                           input(-1).dim(2), 
                                           input(-1).dim(3) }),
                          vector<TIndex>({ input(-1).count(1), 
                                           input(-1).count(2),
                                           input(-1).count(3), 
                                           1 }));
    cudnnSetTensorDesc<T>(&output_desc,
                          vector<TIndex>({ input(0).dim(0), 
                                           input(0).dim(1) / this->group,
                                           input(0).dim(2), 
                                           input(0).dim(3) }),
                          vector<TIndex>({ input(0).count(1), 
                                           input(0).count(2),
                                           input(0).count(3), 
                                           1 }));

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle[0],
                                                          input_desc, 
                                                         output_desc, 
                                                           conv_desc, 
                                                         filter_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                               WORKSPACE_LIMIT_BYTES, 
                                                  &bwd_filter_algo));

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle[0],
                                                              input_desc, 
                                                             output_desc, 
                                                               conv_desc, 
                                                             filter_desc,
                                                         bwd_filter_algo, 
                                            &workspace_bwd_filter_size));

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle[0],
                                                   input_desc, 
                                                  filter_desc, 
                                                    conv_desc, 
                                                  output_desc,
                CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                        WORKSPACE_LIMIT_BYTES, 
                                             &bwd_data_algo));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle[0],
                                                       input_desc, 
                                                      filter_desc, 
                                                        conv_desc, 
                                                      output_desc,
                                                    bwd_data_algo, 
                                       &workspace_bwd_data_size));

    Tensor* buffer1 = ws()->GetBuffer();
    Tensor* buffer2 = ws()->GetBuffer();
    if (workspace_bwd_data_size == 0) workspace_bwd_data_size += 1;
    if (workspace_bwd_filter_size == 0) workspace_bwd_filter_size += 1;
    buffer1->Reshape(vector<TIndex>(1, this->group * workspace_bwd_data_size));
    buffer2->Reshape(vector<TIndex>(1, this->group * workspace_bwd_filter_size));
    if (output(2)->name() != "ignore") {
        bias_offset = this->num_output / this->group;
        cudnnSetTensorDesc<T>(&bias_desc, vector<TIndex>({ 1, bias_offset, 1, 1 }));
    }

    const T* dYdata = input(2).template data<T, Context>();
    for (int g = 0; g < this->group; g++) {
        if (output(2)->name() != "ignore") {
            T* dBdata = output(2)->template mutable_data<T, Context>();
            CUDNN_CHECK(cudnnConvolutionBackwardBias(handle[g],
                                CUDNNType<T>::one, input_desc, dYdata + this->y_offset * g,
                                CUDNNType<T>::one, bias_desc, dBdata + bias_offset * g));
        }
        if (output(1)->name() != "ignore") {
            auto* Xdata = input(0).template data<T, Context>();
            auto* dWdata = output(1)->template mutable_data<T, Context>();
            auto* workspace = buffer2->mutable_data<char, Context>();
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle[1 * this->group + g],
                        CUDNNType<T>::one, input_desc, dYdata + this->y_offset * g,
                                           output_desc, Xdata + this->x_offset * g, 
                                                                         conv_desc, 
                                                                   bwd_filter_algo,
              workspace + g * workspace_bwd_filter_size, workspace_bwd_filter_size,
                CUDNNType<T>::one, filter_desc, dWdata + this->weight_offset * g));
        }
        if (output(0)->name() != "ignore") {
            auto* Wdata = input(1).template data<T, Context>();
            auto* dXdata = output(0)->template mutable_data<T, Context>();
            auto* workspace = buffer1->mutable_data<char, Context>();
            CUDNN_CHECK(cudnnConvolutionForward(handle[2 * this->group + g],
                 CUDNNType<T>::one, input_desc, dYdata + this->y_offset * g,
                               filter_desc, Wdata + this->weight_offset * g, 
                                                                  conv_desc, 
                                                              bwd_data_algo,
           workspace + g * workspace_bwd_data_size, workspace_bwd_data_size,
             CUDNNType<T>::zero, output_desc, dXdata + this->x_offset * g));
        }
    }
    kernel::Empty<T, Context>();
    ws()->ReleaseBuffer(buffer2);
    ws()->ReleaseBuffer(buffer1);
}

template <class Context>
void CuDNNDeConvGradientOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < this->dilation.size(); i++)
        if (this->dilation[i] != 1) return DeConvGradientOp<Context>::RunOnDevice();
#endif
    DeConvGradientOp<Context>::GradientReshape();
    this->x_offset /= this->group;
    this->y_offset /= this->group;

    if (input(0).template IsType<float>()) {
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                         this->dilation[0], this->dilation[1], 
                                      CUDNN_CROSS_CORRELATION, 
                                           CUDNN_DATA_FLOAT));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                                                         1, 1,
                                    CUDNN_CROSS_CORRELATION));
#endif
        RunWithType<float>();
    } else if (input(0).template IsType<float16>()) {
#ifdef WITH_CUDA_FP16
#if CUDNN_VERSION_MIN(6, 0, 0)
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                         this->dilation[0], this->dilation[1], 
                                      CUDNN_CROSS_CORRELATION, 
                                           CUDNN_DATA_FLOAT));
#else
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                   this->pad[0], this->pad[1], 
                             this->stride[0], this->stride[1],
                                                         1, 1, 
                                    CUDNN_CROSS_CORRELATION));
#endif
        RunWithType<float16>();
#endif  // WITH_CUDA_FP16
    } else { LOG(FATAL) << "Unsupported input types."; }
}

DEPLOY_CUDNN(DeConvGradient);

}    // namespace dragon

#endif    // WITH_CUDNN