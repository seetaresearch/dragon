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

#ifndef DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ReluOp : public Operator<Context> {
 public:
  ReluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.f)),
        max_value_(OP_SINGLE_ARG(float, "max_value", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, max_value_;
};

template <class Context>
class ReluGradientOp : public Operator<Context> {
 public:
  ReluGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.f)),
        max_value_(OP_SINGLE_ARG(float, "max_value", 0.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, max_value_;
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNReluOp final : public ReluOp<Context> {
 public:
  CuDNNReluOp(const OperatorDef& def, Workspace* ws)
      : ReluOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    if (this->max_value_ > 0.f) {
      CUDNN_CHECK(cudnnSetActivationDescriptor(
          act_desc_,
          CUDNN_ACTIVATION_CLIPPED_RELU,
          CUDNN_PROPAGATE_NAN,
          this->max_value_));
    } else {
      CUDNN_CHECK(cudnnSetActivationDescriptor(
          act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    }
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNReluOp() {
    CuDNNDestroyTensorDesc(input_desc_);
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    // CuDNN does not support LeakyReLU.
    if (this->alpha_ != 0.f) {
      return ReluOp<Context>::RunOnDevice();
    }
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnActivationDescriptor_t act_desc_;
};

template <class Context>
class CuDNNReluGradientOp final : public ReluGradientOp<Context> {
 public:
  CuDNNReluGradientOp(const OperatorDef& def, Workspace* ws)
      : ReluGradientOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    // CuDNN seems to fallback to the ``ReluGrad`` anyway,
    // even if it set to clipped mode.
    // Use to our kernel implementation temporarily.
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNReluGradientOp() {
    CuDNNDestroyTensorDesc(input_desc_);
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    // CuDNN does not support LeakyReLU and ClippedReLU.
    if (this->alpha_ != 0.f || this->max_value_ > 0.f) {
      return ReluGradientOp<Context>::RunOnDevice();
    }
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnActivationDescriptor_t act_desc_;
};
#endif // USE_CUDNN

#ifdef USE_MPS
template <class Context>
class MPSReluOp final : public Operator<Context> {
 public:
  MPSReluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), graph_(MPSCreateGraph()) {}
  USE_OPERATOR_FUNCTIONS;

  ~MPSReluOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};

template <class Context>
class MPSReluGradientOp final : public Operator<Context> {
 public:
  MPSReluGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), graph_(MPSCreateGraph()) {}
  USE_OPERATOR_FUNCTIONS;

  ~MPSReluGradientOp() {
    NSReleaseObject(graph_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  MPSGraph_t graph_;
  MPSGraphCache graph_cache_;
};
#endif // USE_MPS

#ifdef USE_MLU
template <class Context>
class CNNLReluOp : public Operator<Context> {
 public:
  CNNLReluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.f)),
        max_value_(OP_SINGLE_ARG(float, "max_value", 0.f)) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    auto mode = CNNL_ACTIVATION_RELU;
    auto coef = max_value_ > 0.f ? max_value_ : alpha_;
    if (max_value_ > 0.f) {
      mode = max_value_ == 6.f ? CNNL_ACTIVATION_RELU6
                               : CNNL_ACTIVATION_CLIPPED_RELU;
    } else {
      mode = alpha_ > 0.f ? CNNL_ACTIVATION_LEAKYRELU : CNNL_ACTIVATION_RELU;
    }
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        mode,
        CNNL_ACTIVATION_FAST,
        CNNL_PROPAGATE_NAN,
        coef,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLReluOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, max_value_;
  cnnlTensorDescriptor_t input_desc_;
  cnnlActivationDescriptor_t act_desc_;
};

template <class Context>
class CNNLReluGradientOp final : public CNNLReluOp<Context> {
 public:
  CNNLReluGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLReluOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_RELU_OP_H_
