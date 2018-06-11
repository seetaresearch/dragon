// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_H_

#include "operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class Conv2dOp : public ConvOpBase<Context> {
 public:
    Conv2dOp(const OperatorDef& def, Workspace* ws)
        : ConvOpBase<Context>(def, ws) { 
        this->num_spatial_axes = 2;
        Setup();
    }
    USE_OPERATOR_FUNCTIONS(Context);
    USE_CONVOLUTION_FUNCTIONS(Context);

    bool ReverseDimensions() override { return false; }
    bool HasBias() override { return InputSize() > 2; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class Conv2dGradientOp : public Conv2dOp<Context> {
 public:
    Conv2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Conv2dOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS(Context);
    USE_CONVOLUTION_FUNCTIONS(Context);

    bool HasBias() override { return Output(2)->name() != "ignore"; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNConv2dOp : public Conv2dOp<Context> {
 public:
    CuDNNConv2dOp(const OperatorDef& def, Workspace* ws)
        : Conv2dOp<Context>(def, ws), enable_tensor_core(true) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        cudnn_group = 1;
        enable_tensor_core &= TENSOR_CORE_AVAILABLE();
#else
        cudnn_group = this->group;
        enable_tensor_core = false;
#endif
        handle = new cudnnHandle_t[cudnn_group];
        stream = new cudaStream_t[cudnn_group];
        ctx().SwitchToDevice();
        for (int g = 0; g < cudnn_group; g++) {
            CUDA_CHECK(cudaStreamCreate(&stream[g]));
            CUDNN_CHECK(cudnnCreate(&handle[g]));
            CUDNN_CHECK(cudnnSetStream(handle[g], stream[g]));
        }
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        if (this->data_format == "NCHW") format = CUDNN_TENSOR_NCHW;
        else if (this->data_format == "NHWC") format = CUDNN_TENSOR_NHWC;
        else LOG(FATAL) << "Unknown data format: " << this->data_format;
    }
    USE_OPERATOR_FUNCTIONS(Context);
    USE_CONVOLUTION_FUNCTIONS(Context);

    ~CuDNNConv2dOp() {
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
        for (int g = 0; g < cudnn_group; g++) {
            cudaStreamDestroy(stream[g]);
            CUDNN_CHECK(cudnnDestroy(handle[g]));
        }
    }

    void RunOnDevice() override;
    template <typename T> void ResetDesc();
    template <typename T> void RunWithType();

 protected:
    cudnnHandle_t* handle;
    cudaStream_t*  stream;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t format;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    size_t workspace_fwd_data_size;
    TIndex bias_offset, cudnn_group;
    vector<TIndex> input_dims;
    bool enable_tensor_core;
};

template <class Context>
class CuDNNConv2dGradientOp : public Conv2dGradientOp<Context> {
 public:
    CuDNNConv2dGradientOp(const OperatorDef& def, Workspace* ws)
        : Conv2dGradientOp<Context>(def, ws), enable_tensor_core(true) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        cudnn_group = 1;
        enable_tensor_core &= TENSOR_CORE_AVAILABLE();
#else
        cudnn_group = this->group;
        enable_tensor_core = false;
#endif
        handle = new cudnnHandle_t[cudnn_group * 3];
        stream = new cudaStream_t[cudnn_group * 3];
        for (int g = 0; g < cudnn_group * 3; g++) {
            CUDA_CHECK(cudaStreamCreate(&stream[g]));
            CUDNN_CHECK(cudnnCreate(&handle[g]));
            CUDNN_CHECK(cudnnSetStream(handle[g], stream[g]));
        }
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        if (this->data_format == "NCHW") format = CUDNN_TENSOR_NCHW;
        else if (this->data_format == "NHWC") format = CUDNN_TENSOR_NHWC;
        else LOG(FATAL) << "Unknown data format: " << this->data_format;
    }
    USE_OPERATOR_FUNCTIONS(Context);
    USE_CONVOLUTION_FUNCTIONS(Context);

    ~CuDNNConv2dGradientOp() {
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
        for (int g = 0; g < cudnn_group * 3; g++) {
            cudaStreamDestroy(stream[g]);
            CUDNN_CHECK(cudnnDestroy(handle[g]));
        }
    }

    void RunOnDevice() override;
    template <typename T> void ResetDesc();
    template <typename T> void RunWithType();

 protected:
    cudnnHandle_t* handle;
    cudaStream_t*  stream;
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t format;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    size_t workspace_bwd_filter_size, workspace_bwd_data_size;
    TIndex bias_offset, cudnn_group;
    vector<TIndex> input_dims;
    bool enable_tensor_core;
};

#endif    // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_CONV_OP_H_