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

#ifndef DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SeluOp final : public Operator<Context> {
 public:
  SeluOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.67326f)),
        gamma_(OP_SINGLE_ARG(float, "gamma", 1.0507f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, gamma_;
};

template <class Context>
class SeluGradientOp final : public Operator<Context> {
 public:
  SeluGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 1.67326f)),
        gamma_(OP_SINGLE_ARG(float, "gamma", 1.0507f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float alpha_, gamma_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_SELU_OP_H_
