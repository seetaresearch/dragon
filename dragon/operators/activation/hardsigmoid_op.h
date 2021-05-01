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

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_HARDSIGMOID_OP_H_
