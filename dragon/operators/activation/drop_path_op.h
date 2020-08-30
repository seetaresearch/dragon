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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROPPATH_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROPPATH_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class DropPathOp final : public Operator<Context> {
 public:
  DropPathOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        inc_(OP_SINGLE_ARG(float, "increment", 0.f)) {
    INIT_OP_SINGLE_ARG_WITH_DESC(float, prob, 0.2f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float inc_, drop_prob_ = 0.f;
  DECLARE_OP_SINGLE_ARG_WITH_DESC(float, prob);
};

template <class Context>
class DropPathGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(DropPathGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

DEFINE_OP_SINGLE_ARG_WITH_DESC(float, DropPathOp, prob);

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_DROPPATH_OP_H_
