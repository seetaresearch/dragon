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

#ifndef DRAGON_OPERATORS_LOSS_SOFTMAX_LOSS_OPS_H_
#define DRAGON_OPERATORS_LOSS_SOFTMAX_LOSS_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SoftmaxCrossEntropyOp final : public Operator<Context> {
 public:
  SoftmaxCrossEntropyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OpArg<string>("reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class SparseSoftmaxCrossEntropyOp : public Operator<Context> {
 public:
  SparseSoftmaxCrossEntropyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OpArg<int64_t>("ignore_index", -1)),
        reduction_(OpArg<string>("reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename LogitType, typename TargetType>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
};

template <class Context>
class SoftmaxCrossEntropyGradientOp final : public Operator<Context> {
 public:
  SoftmaxCrossEntropyGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OpArg<string>("reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class SparseSoftmaxCrossEntropyGradientOp : public Operator<Context> {
 public:
  SparseSoftmaxCrossEntropyGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        ignore_index_(OpArg<int64_t>("ignore_index", -1)),
        reduction_(OpArg<string>("reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename LogitType, typename TargetType>
  void DoRunWithType();

 protected:
  int64_t ignore_index_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_SOFTMAX_LOSS_OPS_H_
