/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_H_

#include "operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class Conv2dOp : public ConvOpBase<Context> {
 public:
    Conv2dOp(const OperatorDef& def, Workspace* ws)
        : ConvOpBase<Context>(def, ws) { Setup(2); }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    bool Transposed() override { return false; }
    bool HasBias() override { return XSize() > 2; }

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

template <class Context>
class Conv2dGradientOp
    : public Conv2dOp<Context> {
 public:
    Conv2dGradientOp(
        const OperatorDef&      def,
        Workspace*              ws)
        : Conv2dOp<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    bool HasBias() override {
        return Y(2)->name() != "NULL";
    }

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

#ifdef WITH_CUDNN

template <class Context>
class CuDNNConv2dOp final
    : public Conv2dOp<Context> {
 public:
    CuDNNConv2dOp(
        const OperatorDef&      def,
        Workspace*              ws)
        : Conv2dOp<Context>(def, ws),
          enable_tensor_core_(
            TENSOR_CORE_AVAILABLE() ? 1 : 0
          ) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        cudnn_group_ = 1;
#else
        cudnn_group_ = group_;
#endif
        CuDNNCreateTensorDesc(&input_desc_);
        CuDNNCreateTensorDesc(&bias_desc_);
        CuDNNCreateTensorDesc(&output_desc_);
        CuDNNCreateTensorDesc(&output2b_desc_);
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
        if (data_format() == "NCHW") format_ = CUDNN_TENSOR_NCHW;
        else if (data_format() == "NHWC") format_ = CUDNN_TENSOR_NHWC;
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    ~CuDNNConv2dOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CuDNNDestroyTensorDesc(&bias_desc_);
        CuDNNDestroyTensorDesc(&output_desc_);
        CuDNNDestroyTensorDesc(&output2b_desc_);
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
    }

    void RunOnDevice() override;
    void SetConvDescFromInputs();
    template <typename T> void ResetDesc();
    template <typename T> void RunImpl();

 protected:
    cudnnDataType_t compute_type_;
    cudnnTensorFormat_t format_;
    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnTensorDescriptor_t bias_desc_, output2b_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    size_t fwd_data_size_;
    int64_t cudnn_group_, enable_tensor_core_;
    vec64_t input_dims_, filter_dims_;
};

template <class Context>
class CuDNNConv2dGradientOp final
    : public Conv2dGradientOp<Context> {
 public:
    CuDNNConv2dGradientOp(
        const OperatorDef&      def,
        Workspace*              ws)
        : Conv2dGradientOp<Context>(def, ws),
          enable_tensor_core_(
            TENSOR_CORE_AVAILABLE() ? 1 : 0
          ) {
#if CUDNN_VERSION_MIN(7, 0, 0)
        cudnn_group_ = 1;
#else
        cudnn_group_ = group_;
#endif
        CuDNNCreateTensorDesc(&input_desc_);
        CuDNNCreateTensorDesc(&bias_desc_);
        CuDNNCreateTensorDesc(&output_desc_);
        CuDNNCreateTensorDesc(&input2b_desc_);
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
        if (data_format() == "NCHW") format_ = CUDNN_TENSOR_NCHW;
        else if (data_format() == "NHWC") format_ = CUDNN_TENSOR_NHWC;
    }
    USE_OPERATOR_FUNCTIONS;
    USE_CONVOLUTION_FUNCTIONS;

    ~CuDNNConv2dGradientOp() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CuDNNDestroyTensorDesc(&bias_desc_);
        CuDNNDestroyTensorDesc(&output_desc_);
        CuDNNDestroyTensorDesc(&input2b_desc_);
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
    }

    void SetConvDescFromInputs();

    void RunOnDevice() override;
    template <typename T> void ResetDesc();
    template <typename T> void RunImpl();

 protected:
    cudnnDataType_t compute_type_;
    cudnnTensorFormat_t format_;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnTensorDescriptor_t bias_desc_, input2b_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    size_t bwd_filter_size_, bwd_data_size_;
    int64_t cudnn_group_, enable_tensor_core_;
    vec64_t input_dims_, filter_dims_;
};

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_CONV_OP_H_