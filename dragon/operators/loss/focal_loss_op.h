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

#ifndef DRAGON_OPERATORS_LOSS_FOCAL_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_FOCAL_LOSS_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SigmoidFocalLossOp final : public Operator<Context> {
 public:
  SigmoidFocalLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.25f)),
        gamma_(OP_SINGLE_ARG(float, "gamma", 2.f)),
        start_index_(OP_SINGLE_ARG(int64_t, "start_index", 0)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename InputT, typename TargetT>
  void DoRunWithType();

 protected:
  float alpha_, gamma_;
  int64_t start_index_;
  string reduction_;
};

template <class Context>
class SigmoidFocalLossGradientOp final : public Operator<Context> {
 public:
  SigmoidFocalLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        alpha_(OP_SINGLE_ARG(float, "alpha", 0.25f)),
        gamma_(OP_SINGLE_ARG(float, "gamma", 2.f)),
        start_index_(OP_SINGLE_ARG(int64_t, "start_index", 0)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "VALID")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename InputT, typename TargetT>
  void DoRunWithType();

 protected:
  float alpha_, gamma_;
  int64_t start_index_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_FOCAL_LOSS_OP_H_
