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

#ifndef DRAGON_OPERATORS_NORMALIZATION_LP_NORM_OP_H_
#define DRAGON_OPERATORS_NORMALIZATION_LP_NORM_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class LpNormOp final : public Operator<Context> {
 public:
  LpNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OP_SINGLE_ARG(int64_t, "p", 2)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-12)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t p_;
  double epsilon_;
  string reduction_;
};

template <class Context>
class LpNormGradientOp final : public Operator<Context> {
 public:
  LpNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OP_SINGLE_ARG(int64_t, "p", 2)),
        epsilon_(OP_SINGLE_ARG(double, "epsilon", 1e-12)),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t p_;
  float epsilon_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_NORMALIZATION_LP_NORM_OP_H_
