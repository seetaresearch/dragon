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

#ifndef DRAGON_OPERATORS_LOSS_L1_LOSS_OPS_H_
#define DRAGON_OPERATORS_LOSS_L1_LOSS_OPS_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class L1LossOp final : public Operator<Context> {
 public:
  L1LossOp(const OperatorDef& def, Workspace* ws)
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
class SmoothL1LossOp final : public Operator<Context> {
 public:
  SmoothL1LossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        beta_(OpArg<float>("beta", 1.f)),
        reduction_(OpArg<string>("reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float beta_;
  string reduction_;
};

template <class Context>
class L1LossGradientOp final : public Operator<Context> {
 public:
  L1LossGradientOp(const OperatorDef& def, Workspace* ws)
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
class SmoothL1LossGradientOp final : public Operator<Context> {
 public:
  SmoothL1LossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        beta_(OpArg<float>("beta", 1.f)),
        reduction_(OpArg<string>("reduction", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float beta_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_L1_LOSS_OPS_H_
