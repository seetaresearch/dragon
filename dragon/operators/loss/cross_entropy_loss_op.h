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

#ifndef DRAGON_OPERATORS_LOSS_CROSS_ENTROPY_OP_H_
#define DRAGON_OPERATORS_LOSS_CROSS_ENTROPY_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SigmoidCrossEntropyLossOp final : public Operator<Context> {
 public:
  SigmoidCrossEntropyLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float, double>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class SigmoidCrossEntropyLossGradientOp final : public Operator<Context> {
 public:
  SigmoidCrossEntropyLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float, double>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class SoftmaxCrossEntropyLossOp : public Operator<Context> {
 public:
  SoftmaxCrossEntropyLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OP_SINGLE_ARG(int64_t, "ignore_index", -1)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float, double>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
};

template <class Context>
class SoftmaxCrossEntropyLossGradientOp : public Operator<Context> {
 public:
  SoftmaxCrossEntropyLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OP_SINGLE_ARG(int64_t, "ignore_index", -1)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::TypesBase<float, double>>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_CROSS_ENTROPY_OP_H_
