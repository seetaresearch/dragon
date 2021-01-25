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
class ConvTransposeOp final : public ConvOpBase<Context> {
 public:
  ConvTransposeOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool Transposed() override {
    return true;
  }

  bool HasBias() override {
    return InputSize() > 2;
  }
};

template <class Context>
class ConvTransposeGradientOp final : public ConvOpBase<Context> {
 public:
  ConvTransposeGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool Transposed() override {
    return true;
  }

  bool HasBias() override {
    return Output(2)->has_name();
  }
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNConvTransposeOp final : public CuDNNConvOpBase<Context> {
 public:
  CuDNNConvTransposeOp(const OperatorDef& def, Workspace* ws)
      : CuDNNConvOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CuDNNCreateTensorDesc(&output_desc_for_bias_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_CUDNN_CONV_FUNCTIONS;

  ~CuDNNConvTransposeOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CuDNNDestroyTensorDesc(&output_desc_for_bias_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }

  bool Transposed() override {
    return true;
  }

  template <typename T>
  void ResetDesc();

  size_t cudnn_ws_size_;
  vec64_t input_dims_, filter_dims_;
  bool exhaustive_search_ = false;
  bool algo_deterministic_ = false;
  cudnnConvolutionBwdDataAlgo_t fwd_algo_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bias_desc_, output_desc_for_bias_;
  using FwdAlgoWithCost = std::tuple<cudnnConvolutionBwdDataAlgo_t, float>;
  ConvAlgorithmCache<FwdAlgoWithCost> algo_cache_;
};

template <class Context>
class CuDNNConvTransposeGradientOp final : public CuDNNConvOpBase<Context> {
 public:
  CuDNNConvTransposeGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNConvOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&bias_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CuDNNCreateTensorDesc(&input_desc_for_bias_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_CUDNN_CONV_FUNCTIONS;

  ~CuDNNConvTransposeGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&bias_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CuDNNDestroyTensorDesc(&input_desc_for_bias_);
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }

  bool Transposed() override {
    return true;
  }

  template <typename T>
  void ResetDesc();

  size_t cudnn_ws_size_;
  vec64_t input_dims_, filter_dims_;
  bool exhaustive_search_data_ = false;
  bool exhaustive_search_filter_ = false;
  bool data_algo_deterministic_ = false;
  bool filter_algo_deterministic_ = false;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionFwdAlgo_t bwd_data_algo_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bias_desc_, input_desc_for_bias_;
  using BwdDataAlgoWithCost = std::tuple<cudnnConvolutionFwdAlgo_t, float>;
  using BwdFilterAlgoWithCost =
      std::tuple<cudnnConvolutionBwdFilterAlgo_t, float>;
  ConvAlgorithmCache<BwdDataAlgoWithCost> data_algo_cache_;
  ConvAlgorithmCache<BwdFilterAlgoWithCost> filter_algo_cache_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_
