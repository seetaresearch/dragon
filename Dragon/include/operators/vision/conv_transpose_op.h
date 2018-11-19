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

#ifndef DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_
#define DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_

#include "operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class Conv2dTransposeOp : public ConvOpBase<Context> {
 public:
    Conv2dTransposeOp(const OperatorDef& def, Workspace* ws) 
        : ConvOpBase<Context>(def, ws) {
        this->num_spatial_axes = 2;
        Setup(); 
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS(Context);

    bool ReverseDimensions() override { return true; }
    bool HasBias() override { return InputSize() > 2; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int> static_dsize;
    vector<string> dynamic_dsize;
};

template <class Context>
class Conv2dTransposeGradientOp : public Conv2dTransposeOp<Context> {
 public:
    Conv2dTransposeGradientOp(const OperatorDef& def, Workspace* ws)
        : Conv2dTransposeOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS(Context);

    bool HasBias() override { return Output(2)->name() != "ignore"; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNConv2dTransposeOp final : public Conv2dTransposeOp<Context> {
 public:
    CuDNNConv2dTransposeOp(const OperatorDef& def, Workspace* ws)
        : Conv2dTransposeOp<Context>(def, ws), enable_tensor_core(true) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        cudnn_group = 1;
        enable_tensor_core &= TENSOR_CORE_AVAILABLE();
#else
        cudnn_group = group;
        enable_tensor_core = false;
#endif
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        if (data_format == "NCHW") format = CUDNN_TENSOR_NCHW;
        else if (data_format == "NHWC") format = CUDNN_TENSOR_NHWC;
        else LOG(FATAL) << "Unknown data format: " << data_format;
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS(Context);

    ~CuDNNConv2dTransposeOp() {
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    void RunOnDevice() override;
    template <typename T> void ResetDesc();
    template <typename T> void RunWithType();

 protected:
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t format;
    cudnnConvolutionBwdDataAlgo_t fwd_algo;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    size_t fwd_data_size;
    TIndex cudnn_group;
    vector<TIndex> input_dims;
    bool enable_tensor_core;
};

template <class Context>
class CuDNNConv2dTransposeGradientOp final : public Conv2dTransposeGradientOp<Context> {
public:
    CuDNNConv2dTransposeGradientOp(const OperatorDef& def, Workspace* ws)
        : Conv2dTransposeGradientOp<Context>(def, ws), enable_tensor_core(true) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        cudnn_group = 1;
        enable_tensor_core &= TENSOR_CORE_AVAILABLE();
#else
        cudnn_group = group;
        enable_tensor_core = false;
#endif
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        if (data_format == "NCHW") format = CUDNN_TENSOR_NCHW;
        else if (data_format == "NHWC") format = CUDNN_TENSOR_NHWC;
        else LOG(FATAL) << "Unknown data format: " << data_format;
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS(Context);

    ~CuDNNConv2dTransposeGradientOp() {
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        if (HasBias()) CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    void RunOnDevice() override;
    template <typename T> void ResetDesc();
    template <typename T> void RunWithType();

 protected:
    cudnnDataType_t compute_type;
    cudnnTensorFormat_t format;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionFwdAlgo_t bwd_data_algo;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    size_t bwd_filter_size, bwd_data_size;
    TIndex cudnn_group;
    vector<TIndex> input_dims;
    bool enable_tensor_core;
};

#endif    // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_