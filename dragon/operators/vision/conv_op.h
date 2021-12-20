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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_H_

#include "dragon/operators/vision/conv_op_algo.h"
#include "dragon/operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
class ConvOp final : public ConvOpBase<Context> {
 public:
  explicit ConvOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }
};

template <class Context>
class ConvGradientOp final : public ConvOpBase<Context> {
 public:
  ConvGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNConvOp final : public CuDNNConvOpBase<Context> {
 public:
  CuDNNConvOp(const OperatorDef& def, Workspace* ws)
      : CuDNNConvOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CuDNNCreateTensorDesc(&bias_desc_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_CUDNN_CONV_FUNCTIONS;

  ~CuDNNConvOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CuDNNDestroyTensorDesc(&bias_desc_);
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

  template <typename T>
  void SetOpDesc();

  vec64_t input_dims_, filter_dims_;
  cudnnTensorDescriptor_t input_desc_, output_desc_, bias_desc_;

  using FwdAlgo = cudnnConvolutionFwdAlgo_t;
  using FwdAlgoWithCost = std::tuple<FwdAlgo, float>;
  FwdAlgo fwd_algo_;
  ConvAlgoCache<FwdAlgoWithCost> fwd_algo_cache_;
  bool exhaustive_search_;
};

template <class Context>
class CuDNNConvGradientOp final : public CuDNNConvOpBase<Context> {
 public:
  CuDNNConvGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNConvOpBase<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CuDNNCreateTensorDesc(&output_desc_);
    CuDNNCreateTensorDesc(&bias_desc_);
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_CUDNN_CONV_FUNCTIONS;

  ~CuDNNConvGradientOp() {
    CuDNNDestroyTensorDesc(&input_desc_);
    CuDNNDestroyTensorDesc(&output_desc_);
    CuDNNDestroyTensorDesc(&bias_desc_);
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

  template <typename T>
  void SetOpDesc();

  vec64_t input_dims_, filter_dims_;
  cudnnTensorDescriptor_t input_desc_, output_desc_, bias_desc_;

  using BwdDataAlgo = cudnnConvolutionBwdDataAlgo_t;
  using BwdFilterAlgo = cudnnConvolutionBwdFilterAlgo_t;
  using BwdDataAlgoWithCost = std::tuple<BwdDataAlgo, float>;
  using BwdFilterAlgoWithCost = std::tuple<BwdFilterAlgo, float>;
  BwdDataAlgo bwd_data_algo_;
  BwdFilterAlgo bwd_filter_algo_;
  ConvAlgoCache<BwdDataAlgoWithCost> data_algo_cache_;
  ConvAlgoCache<BwdFilterAlgoWithCost> filter_algo_cache_;
  bool exhaustive_search_data_, exhaustive_search_filter_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_H_
