/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_
#define DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_

#include "dragon/operators/vision/conv_op_base.h"
#include "dragon/operators/vision/conv_op_cache.h"

namespace dragon {

template <class Context>
class ConvTranspose2dOp : public ConvOpBase<Context> {
 public:
  ConvTranspose2dOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    Setup(2);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  bool Transposed() override {
    return true;
  }

  bool HasBias() override {
    return InputSize() > 2;
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class ConvTranspose2dGradientOp : public ConvTranspose2dOp<Context> {
 public:
  ConvTranspose2dGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvTranspose2dOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  bool HasBias() override {
    return Output(2)->has_name();
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNConvTranspose2dOp final : public ConvTranspose2dOp<Context> {
 public:
  CuDNNConvTranspose2dOp(const OperatorDef& def, Workspace* ws)
      : ConvTranspose2dOp<Context>(def, ws),
        enable_tensor_core_(TENSOR_CORE_AVAILABLE() ? 1 : 0) {
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
    if (data_format() == "NCHW") {
      format_ = CUDNN_TENSOR_NCHW;
    } else if (data_format() == "NHWC") {
      format_ = CUDNN_TENSOR_NHWC;
    }
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  ~CuDNNConvTranspose2dOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CuDNNDestroyTensorDesc(&output2b_desc_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void SetConvDesc();

  template <typename T>
  void ResetDesc();

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnDataType_t compute_type_;
  cudnnTensorFormat_t format_;
  cudnnConvolutionBwdDataAlgo_t fwd_algo_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bias_desc_, output2b_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  size_t cudnn_ws_nbytes_;
  vec64_t input_dims_, filter_dims_;
  int64_t cudnn_group_, enable_tensor_core_;
  bool exhaustive_search_ = false;
  using FwdAlgoWithCost = std::tuple<cudnnConvolutionBwdDataAlgo_t, float>;
  ConvAlgorithmCache<FwdAlgoWithCost> algo_cache_;
};

template <class Context>
class CuDNNConvTranspose2dGradientOp final
    : public ConvTranspose2dGradientOp<Context> {
 public:
  CuDNNConvTranspose2dGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvTranspose2dGradientOp<Context>(def, ws),
        enable_tensor_core_(TENSOR_CORE_AVAILABLE() ? 1 : 0) {
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
    if (data_format() == "NCHW")
      format_ = CUDNN_TENSOR_NCHW;
    else if (data_format() == "NHWC")
      format_ = CUDNN_TENSOR_NHWC;
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONVOLUTION_FUNCTIONS;

  ~CuDNNConvTranspose2dGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CuDNNDestroyTensorDesc(&input2b_desc_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void SetConvDesc();

  template <typename T>
  void ResetDesc();

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnDataType_t compute_type_;
  cudnnTensorFormat_t format_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionFwdAlgo_t bwd_data_algo_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bias_desc_, input2b_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  size_t cudnn_ws_nbytes_;
  vec64_t input_dims_, filter_dims_;
  int64_t cudnn_group_, enable_tensor_core_;
  bool exhaustive_search_data_ = false;
  bool exhaustive_search_filter_ = false;
  using BwdDataAlgoWithCost = std::tuple<cudnnConvolutionFwdAlgo_t, float>;
  using BwdFilterAlgoWithCost =
      std::tuple<cudnnConvolutionBwdFilterAlgo_t, float>;
  ConvAlgorithmCache<BwdDataAlgoWithCost> data_algo_cache_;
  ConvAlgorithmCache<BwdFilterAlgoWithCost> filter_algo_cache_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_
