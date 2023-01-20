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

#include "dragon/operators/math/reduce_op_impl_cnnl.h"
#include "dragon/operators/vision/conv_op_algo.h"
#include "dragon/operators/vision/conv_op_base.h"
#include "dragon/operators/vision/conv_op_impl_cnnl.h"
#include "dragon/operators/vision/conv_op_impl_cudnn.h"

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

  void RunOnDevice() override {
    if (data_format() == "NHWC" && group_ != 1) {
      LOG(FATAL) << "GroupConv(NHWC) is not supported.";
    }
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

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

  void RunOnDevice() override {
    if (data_format() == "NHWC" && group_ != 1) {
      LOG(FATAL) << "GroupConv(NHWC) is not supported.";
    }
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNConvOp final : public ConvOpBase<Context> {
 public:
  CuDNNConvOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }

  CuDNNConvOpImpl<cudnnConvolutionFwdAlgo_t> Y_impl_;
};

template <class Context>
class CuDNNConvGradientOp final : public ConvOpBase<Context> {
 public:
  CuDNNConvGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }

  CuDNNConvOpImpl<cudnnConvolutionBwdDataAlgo_t> dX_impl_;
  CuDNNConvOpImpl<cudnnConvolutionBwdFilterAlgo_t> dW_impl_;
};
#endif // USE_CUDNN

#ifdef USE_MPS
template <class Context>
class MPSConvOp final : public MPSConvOpBase<Context> {
 public:
  MPSConvOp(const OperatorDef& def, Workspace* ws)
      : MPSConvOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
    SetConvDesc();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_MPS_CONV_FUNCTIONS;

  ~MPSConvOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }

  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSConvGradientOp final : public MPSConvOpBase<Context> {
 public:
  MPSConvGradientOp(const OperatorDef& def, Workspace* ws)
      : MPSConvOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
    SetConvDesc();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_MPS_CONV_FUNCTIONS;

  ~MPSConvGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }

  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLConvOp final : public ConvOpBase<Context> {
 public:
  CNNLConvOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    CHECK_EQ(data_format(), "NHWC");
    GetBaseArguments();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }

  CNNLConvOpImpl<cnnlConvolutionForwardAlgo_t> Y_impl_;
};

template <class Context>
class CNNLConvGradientOp final : public ConvOpBase<Context> {
 public:
  CNNLConvGradientOp(const OperatorDef& def, Workspace* ws)
      : ConvOpBase<Context>(def, ws) {
    CHECK_EQ(data_format(), "NHWC");
    GetBaseArguments();
    dB_impl_.SetReducer(CNNL_REDUCE_ADD);
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }

  CNNLConvOpImpl<cnnlConvolutionBwdDataAlgo_t> dX_impl_;
  CNNLConvOpImpl<cnnlConvolutionBwdFilterAlgo_t> dW_impl_;
  CNNLReduceOpImpl dB_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_H_
