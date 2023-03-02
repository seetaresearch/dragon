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

#ifndef DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class EluOp : public Operator<Context> {
 public:
  EluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_;
};

template <class Context>
class EluGradientOp : public Operator<Context> {
 public:
  EluGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_;
};

#ifdef USE_CUDNN
template <class Context>
class CuDNNEluOp final : public EluOp<Context> {
 public:
  CuDNNEluOp(const OperatorDef& def, Workspace* ws) : EluOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_, CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, this->alpha_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNEluOp() {
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
class CuDNNEluGradientOp final : public EluGradientOp<Context> {
 public:
  CuDNNEluGradientOp(const OperatorDef& def, Workspace* ws)
      : EluGradientOp<Context>(def, ws) {
    CuDNNCreateTensorDesc(&input_desc_);
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_, CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, this->alpha_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNEluGradientOp() {
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
#endif // USE_CUDNN

#ifdef USE_MLU
template <class Context>
class CNNLEluOp : public Operator<Context> {
 public:
  CNNLEluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), alpha_(OP_SINGLE_ARG(float, "alpha", 1.f)) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_ELU,
        CNNL_ACTIVATION_FAST,
        CNNL_NOT_PROPAGATE_NAN,
        alpha_,
        0,
        1.f, // gamma
        1.f, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLEluOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_;
  cnnlTensorDescriptor_t input_desc_;
  cnnlActivationDescriptor_t act_desc_;
};

template <class Context>
class CNNLEluGradientOp final : public CNNLEluOp<Context> {
 public:
  CNNLEluGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLEluOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_ELU_OP_H_
