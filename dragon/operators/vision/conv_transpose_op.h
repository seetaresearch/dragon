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

#include "dragon/operators/math/reduce_op_impl_cnnl.h"
#include "dragon/operators/vision/conv_op_algo.h"
#include "dragon/operators/vision/conv_op_base.h"
#include "dragon/operators/vision/conv_op_impl_cnnl.h"
#include "dragon/operators/vision/conv_op_impl_cudnn.h"

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

  void RunOnDevice() override {
    if (data_format() == "NHWC" && group_ != 1) {
      LOG(FATAL) << "GroupConvNHWC is not supported.";
    }
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return InputSize() > 2;
  }

  bool Transposed() override {
    return true;
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

  void RunOnDevice() override {
    if (data_format() == "NHWC" && group_ != 1) {
      LOG(FATAL) << "GroupConvNHWC is not supported.";
    }
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  bool HasBias() override {
    return Output(2)->has_name();
  }

  bool Transposed() override {
    return true;
  }
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNConvTransposeOp final : public ConvOpBase<Context> {
 public:
  CuDNNConvTransposeOp(const OperatorDef& def, Workspace* ws)
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

  bool Transposed() override {
    return true;
  }

  CuDNNConvOpImpl<cudnnConvolutionBwdDataAlgo_t> Y_impl_;
};

template <class Context>
class CuDNNConvTransposeGradientOp final : public ConvOpBase<Context> {
 public:
  CuDNNConvTransposeGradientOp(const OperatorDef& def, Workspace* ws)
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

  bool Transposed() override {
    return true;
  }

  CuDNNConvOpImpl<cudnnConvolutionFwdAlgo_t> dX_impl_;
  CuDNNConvOpImpl<cudnnConvolutionBwdFilterAlgo_t> dW_impl_;
};
#endif // USE_CUDNN

#ifdef USE_MPS
template <class Context>
class MPSConvTransposeOp final : public MPSConvOpBase<Context> {
 public:
  MPSConvTransposeOp(const OperatorDef& def, Workspace* ws)
      : MPSConvOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
    SetConvDesc();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_MPS_CONV_FUNCTIONS;

  ~MPSConvTransposeOp() {
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

  bool Transposed() override {
    return true;
  }

  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSConvTransposeGradientOp final : public MPSConvOpBase<Context> {
 public:
  MPSConvTransposeGradientOp(const OperatorDef& def, Workspace* ws)
      : MPSConvOpBase<Context>(def, ws) {
    graph_ = MPSCreateGraph();
    SetConvDesc();
  }
  USE_OPERATOR_FUNCTIONS;
  USE_CONV_FUNCTIONS;
  USE_MPS_CONV_FUNCTIONS;

  ~MPSConvTransposeGradientOp() {
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

  bool Transposed() override {
    return true;
  }

  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLConvTransposeOp final : public ConvOpBase<Context> {
 public:
  CNNLConvTransposeOp(const OperatorDef& def, Workspace* ws)
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

  bool Transposed() override {
    return true;
  }

  CNNLDeconvOpImpl<cnnlDeconvolutionAlgo_t> Y_impl_;
};

template <class Context>
class CNNLConvTransposeGradientOp final : public ConvOpBase<Context> {
 public:
  CNNLConvTransposeGradientOp(const OperatorDef& def, Workspace* ws)
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

  bool Transposed() override {
    return true;
  }

  CNNLConvOpImpl<cnnlConvolutionForwardAlgo_t> dX_impl_;
  CNNLConvOpImpl<cnnlConvolutionBwdFilterAlgo_t> dW_impl_;
  CNNLReduceOpImpl dB_impl_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_TRANSPOSE_OP_H_
