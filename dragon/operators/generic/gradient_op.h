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

#ifndef DRAGON_OPERATORS_GENERIC_GRADIENT_OP_H_
#define DRAGON_OPERATORS_GENERIC_GRADIENT_OP_H_

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

  void RunOnDevice() override {
    inputs_.clear();
    auto* Y = Output(0);
    for (int i = 0; i < InputSize(); ++i) {
      auto* X = &Input(i);
      if (X->has_name()) inputs_.push_back(X);
    }
    if (inputs_.empty() || !Y->has_name()) return;
    DispatchHelper<dtypes::Floating>::Call(this, *inputs_[0]);
  }

  template <typename T>
  void DoRunWithType();

 protected:
  vector<Tensor*> inputs_;
};

template <class Context>
class GradientStopOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(GradientStopOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    auto &X = Input(0), *Y = Output(0, {0});
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  }
};

} // namespace dragon

#endif // DRAGON_OPERATORS_GENERIC_GRADIENT_OP_H_
