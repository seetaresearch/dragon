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

#ifndef DRAGON_OPERATORS_ACTIVATION_HARDSIGMOID_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_HARDSIGMOID_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class HardSigmoidOp : public Operator<Context> {
 public:
  HardSigmoidOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.2f)),
        beta_(OP_SINGLE_ARG(float, "beta", 0.5f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
};

template <class Context>
class HardSigmoidGradientOp : public Operator<Context> {
 public:
  HardSigmoidGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.2f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_;
};

#ifdef USE_MLU
template <class Context>
class CNNLHardSigmoidOp : public Operator<Context> {
 public:
  CNNLHardSigmoidOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.2)),
        beta_(OP_SINGLE_ARG(float, "beta", 0.5f)) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNL_CHECK(cnnlCreateActivationDescriptor(&act_desc_));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(
        act_desc_,
        CNNL_ACTIVATION_HARDSIGMOID,
        CNNL_ACTIVATION_FAST,
        CNNL_NOT_PROPAGATE_NAN,
        0.f,
        0,
        alpha_, // gamma
        beta_, // scale
        true,
        false));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLHardSigmoidOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNL_CHECK(cnnlDestroyActivationDescriptor(act_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
  cnnlTensorDescriptor_t input_desc_;
  cnnlActivationDescriptor_t act_desc_;
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_HARDSIGMOID_OP_H_
