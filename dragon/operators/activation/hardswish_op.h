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

#ifndef DRAGON_OPERATORS_ACTIVATION_HARDSWISH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_HARDSWISH_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class HardSwishOp : public Operator<Context> {
 public:
  HardSwishOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.2f)),
        beta_(OP_SINGLE_ARG(float, "beta", 0.5f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
};

template <class Context>
class HardSwishGradientOp : public Operator<Context> {
 public:
  HardSwishGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.2f)),
        beta_(OP_SINGLE_ARG(float, "beta", 0.5f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, beta_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_HARDSWISH_OP_H_
