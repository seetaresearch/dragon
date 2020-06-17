/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_NORMALIZATION_LP_NORMALIZE_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_LP_NORMALIZE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class LpNormalizeOp final : public Operator<Context> {
 public:
  LpNormalizeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OpArg<int64_t>("p", 2)),
        eps_(OpArg<float>("eps", 1e-12f)),
        reduction_(OpArg<string>("reduction", "SUM")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float eps_;
  string reduction_;
  int64_t p_;
};

template <class Context>
class LpNormalizeGradientOp final : public Operator<Context> {
 public:
  LpNormalizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OpArg<int64_t>("p", 2)),
        eps_(OpArg<float>("eps", 1e-12f)),
        reduction_(OpArg<string>("reduction", "SUM")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float eps_;
  string reduction_;
  int64_t p_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_LP_NORMALIZE_OP_H_
