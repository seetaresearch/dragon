#ifdef WITH_CUDNN

#include "operators/vision/conv_transpose_op.h"
#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"

namespace dragon {

#define WORKSPACE_LIMIT_BYTES 64 * 1024 * 1024 // 64MB

template <class Context> template <typename T>
void CuDNNConv2dTransposeOp<Context>::RunWithType() {
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc,
                                    CUDNNType<T>::type,
                                                format,
                        this->num_output / cudnn_group,
                          this->channels / this->group,
          this->kernel_size[0], this->kernel_size[1]));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc,
                                       CUDNNType<T>::type,
                                                   format,
                           this->num_output / cudnn_group,
                             this->channels / this->group,
             this->kernel_size[0], this->kernel_size[1]));
#endif

    //  determine the input & output shape
    cudnnSetTensor4dDescWithGroup<T>(&input_desc, this->data_format, input(0).dims(), cudnn_group);
    cudnnSetTensor4dDescWithGroup<T>(&output_desc, this->data_format, output(0)->dims(), cudnn_group);

    //  determine the bias shape
    if (HasBias()) {
        bias_offset = this->num_output / cudnn_group;
        if (this->data_format == "NCHW") {
            cudnnSetTensor4dDesc<T>(&bias_desc, this->data_format, vector<TIndex>({ 1, bias_offset, 1, 1 }));
        } else if (this->data_format == "NHWC") {
            cudnnSetTensor4dDesc<T>(&bias_desc, this->data_format, vector<TIndex>({ 1, 1, 1, bias_offset }));
        }
    }

    //  determine the misc
    if (HasBias()) {
        if (this->data_format == "NCHW") {
            this->x_offset = input(0).count(1) / cudnn_group;
            this->y_offset = output(0)->count(1) / cudnn_group;
        } else if (this->data_format == "NHWC") {
            this->x_offset = input(0).dim(-1) / cudnn_group;
            this->y_offset = output(0)->dim(-1) / cudnn_group;
        }
    }

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
    buffer->Reshape(vector<TIndex>(1, cudnn_group * workspace_fwd_data_size));

    auto* Xdata = input(0).template data<T, Context>();
    auto* Ydata = output(0)->template mutable_data<T, Context>();
    TENSOR_FILL(input(1), this->weight_shape);
    auto* Wdata = input(1).template data<T, Context>();
    if (HasBias()) TENSOR_FILL(input(2), this->bias_shape);

    for (int g = 0; g < cudnn_group; g++) {
        auto* workspace = buffer->template mutable_data<char, Context>();
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle[g],
                        CUDNNType<T>::one, filter_desc, Wdata + this->weight_offset * g,
                                                 input_desc, Xdata + this->x_offset * g,
                                                                              conv_desc,
                                                                               fwd_algo,
                       workspace + g * workspace_fwd_data_size, workspace_fwd_data_size,
                          CUDNNType<T>::zero, output_desc, Ydata + this->y_offset * g));

        if (HasBias()) {
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
void CuDNNConv2dTransposeOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < this->dilation.size(); i++)
        if (this->dilation[i] != 1) return Conv2dTransposeOp<Context>::RunOnDevice();
#endif
    Conv2dTransposeOp<Context>::Reshape();

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
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, this->group));
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
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, this->group));
#endif
        RunWithType<float16>();
#endif  // WITH_CUDA_FP16
    } else { LOG(FATAL) << "Unsupported input types."; }
}

DEPLOY_CUDNN(Conv2dTranspose);

template <class Context> template <typename T>
void CuDNNConv2dTransposeGradientOp<Context>::RunWithType() {
#if CUDNN_VERSION_MIN(5, 0, 0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc,
                                    CUDNNType<T>::type,
                                                format,
                        this->num_output / cudnn_group,
                          this->channels / this->group,
          this->kernel_size[0], this->kernel_size[1]));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(filter_desc,
                                       CUDNNType<T>::type,
                                                   format,
                           this->num_output / cudnn_group,
                             this->channels / this->group,
             this->kernel_size[0], this->kernel_size[1]));
#endif

    //  determine the input & output shape
    cudnnSetTensor4dDescWithGroup<T>(&input_desc, this->data_format, input(-1).dims(), cudnn_group);
    cudnnSetTensor4dDescWithGroup<T>(&output_desc, this->data_format, input(0).dims(), cudnn_group);

    //  determine the bias shape
    if (HasBias()) {
        bias_offset = this->num_output / cudnn_group;
        if (this->data_format == "NCHW") {
            cudnnSetTensor4dDesc<T>(&bias_desc, this->data_format, vector<TIndex>({ 1, bias_offset, 1, 1 }));
        } else if (this->data_format == "NHWC") {
            cudnnSetTensor4dDesc<T>(&bias_desc, this->data_format, vector<TIndex>({ 1, 1, 1, bias_offset }));
        }
    }

    //  determine the misc
    if (HasBias()) {
        if (this->data_format == "NCHW") {
            this->x_offset = input(0).count(1) / cudnn_group;
            this->y_offset = input(-1).count(1) / cudnn_group;
        } else if (this->data_format == "NHWC") {
            this->x_offset = input(0).dim(-1) / cudnn_group;
            this->y_offset = input(-1).dim(-1) / cudnn_group;
        }
    }

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
    buffer1->Reshape(vector<TIndex>(1, cudnn_group * workspace_bwd_data_size));
    buffer2->Reshape(vector<TIndex>(1, cudnn_group * workspace_bwd_filter_size));

    const T* dYdata = input(2).template data<T, Context>();
    for (int g = 0; g < cudnn_group; g++) {
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
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle[1 * cudnn_group + g],
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
            CUDNN_CHECK(cudnnConvolutionForward(handle[2 * cudnn_group + g],
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
void CuDNNConv2dTransposeGradientOp<Context>::RunOnDevice() {
#if CUDNN_VERSION_MAX(6, 0, 0)
    for (int i = 0; i < this->dilation.size(); i++)
        if (this->dilation[i] != 1) return Conv2dTransposeGradientOp<Context>::RunOnDevice();
#endif
    Conv2dTransposeGradientOp<Context>::GradientReshape();

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
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, this->group));
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
#if CUDNN_VERSION_MIN(7, 0, 0)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, this->group));
#endif
        RunWithType<float16>();
#endif  // WITH_CUDA_FP16
    } else { LOG(FATAL) << "Unsupported input types."; }
}

DEPLOY_CUDNN(Conv2dTransposeGradient);

}    // namespace dragon

#endif    // WITH_CUDNN
