// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_
#define DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_

#include "operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class Conv2dTransposeOp: public ConvOpBase<Context>  {
 public:
    Conv2dTransposeOp(const OperatorDef& def, Workspace* ws) 
        : ConvOpBase<Context>(def, ws) {
        this->num_spatial_axes = 2;
        Setup(); 
    }

    bool ReverseDimensions() override { return true; }
    virtual bool HasBias() { return InputSize() > 2; }

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

    bool HasBias() override { return output(2)->name() != "ignore"; }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNConv2dTransposeOp : public Conv2dTransposeOp<Context> {
 public:
    CuDNNConv2dTransposeOp(const OperatorDef& def, Workspace* ws)
        : Conv2dTransposeOp<Context>(def, ws) {
        handle = new cudnnHandle_t[this->group];
        stream = new cudaStream_t[this->group];
        for (int g = 0; g < this->group; g++) {
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
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnHandle_t* handle;
    cudaStream_t*  stream;
    cudnnTensorFormat_t format;
    cudnnConvolutionBwdDataAlgo_t fwd_algo;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    size_t workspace_fwd_data_size;
    int bias_offset;
};

template <class Context>
class CuDNNConv2dTransposeGradientOp : public Conv2dTransposeGradientOp<Context> {
public:
    CuDNNConv2dTransposeGradientOp(const OperatorDef& def, Workspace* ws)
        : Conv2dTransposeGradientOp<Context>(def, ws) {
        handle = new cudnnHandle_t[this->group * 3];
        stream = new cudaStream_t[this->group * 3];
        for (int g = 0; g < this->group * 3; g++) {
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
    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnHandle_t* handle;
    cudaStream_t*  stream;
    cudnnTensorFormat_t format;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionFwdAlgo_t bwd_data_algo;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;
    size_t workspace_bwd_filter_size, workspace_bwd_data_size;
    int bias_offset;
};

#endif    // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_