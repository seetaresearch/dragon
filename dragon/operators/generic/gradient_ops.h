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

#ifndef DRAGON_OPERATORS_GENERIC_GRADIENT_OPS_H_
#define DRAGON_OPERATORS_GENERIC_GRADIENT_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class GradientFillOp final : public Operator<Context> {
 public:
  GradientFillOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), values_(OP_REPEATED_ARG(float, "values")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vector<float> values_;
};

template <class Context>
class GradientGatherOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GradientGatherOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  vector<Tensor*> grads_;
};

template <class Context>
class GradientAddOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GradientAddOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

template <class Context>
class StopGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(StopGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_GENERIC_GRADIENT_OPS_H_
