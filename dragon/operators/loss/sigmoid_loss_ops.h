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

#ifndef DRAGON_OPERATORS_LOSS_SIGMOID_LOSS_OPS_H_
#define DRAGON_OPERATORS_LOSS_SIGMOID_LOSS_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SigmoidCrossEntropyOp final : public Operator<Context> {
 public:
  SigmoidCrossEntropyOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OpArg<string>("reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class SigmoidFocalLossOp final : public Operator<Context> {
 public:
  SigmoidFocalLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        pos_alpha_(OpArg<float>("alpha", 0.25f)),
        neg_alpha_(1.f - OpArg<float>("alpha", 0.25f)),
        gamma_(OpArg<float>("gamma", 2.f)),
        negative_index_(OpArg<int64_t>("negative_index", -1)),
        reduction_(OpArg<string>("reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename LogitType, typename TargetType>
  void DoRunWithType();

 protected:
  int64_t negative_index_;
  float pos_alpha_, neg_alpha_, gamma_;
  string reduction_;
};

template <class Context>
class SigmoidCrossEntropyGradientOp final : public Operator<Context> {
 public:
  SigmoidCrossEntropyGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        reduction_(OpArg<string>("reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string reduction_;
};

template <class Context>
class SigmoidFocalLossGradientOp final : public Operator<Context> {
 public:
  SigmoidFocalLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        pos_alpha_(OpArg<float>("alpha", 0.25f)),
        neg_alpha_(1.f - OpArg<float>("alpha", 0.25f)),
        gamma_(OpArg<float>("gamma", 2.f)),
        negative_index_(OpArg<int64_t>("negative_index", -1)),
        reduction_(OpArg<string>("reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename LogitType, typename TargetType>
  void DoRunWithType();

 protected:
  int64_t negative_index_;
  float gamma_, pos_alpha_, neg_alpha_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_SIGMOID_LOSS_OPS_H_
