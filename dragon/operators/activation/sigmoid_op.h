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

#ifndef DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SigmoidOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SigmoidOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class SigmoidGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(SigmoidGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNSigmoidOp : public Operator<Context> {
 public:
  CuDNNSigmoidOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNSigmoidOp() {
    CuDNNDestroyTensorDesc(input_desc_);
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cudnnTensorDescriptor_t input_desc_;
  cudnnActivationDescriptor_t act_desc_;
};

template <class Context>
class CuDNNSigmoidGradientOp final : public CuDNNSigmoidOp<Context> {
 public:
  CuDNNSigmoidGradientOp(const OperatorDef& def, Workspace* ws)
      : CuDNNSigmoidOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_CUDNN

#ifdef USE_MLU
template <class Context>
class CNNLSigmoidOp : public Operator<Context> {
 public:
  CNNLSigmoidOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_SIGMOID,
        CNNL_ACTIVATION_HIGH_PRECISION,
        CNNL_PROPAGATE_NAN,
        0.f,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLSigmoidOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  cnnlTensorDescriptor_t input_desc_;
  cnnlActivationDescriptor_t act_desc_;
};

template <class Context>
class CNNLSigmoidGradientOp final : public CNNLSigmoidOp<Context> {
 public:
  CNNLSigmoidGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLSigmoidOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_SIGMOID_OP_H_
