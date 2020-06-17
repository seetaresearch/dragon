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

#ifndef DRAGON_OPERATORS_ACTIVATION_DROP_BLOCK_OP_H_
#define DRAGON_OPERATORS_ACTIVATION_DROP_BLOCK_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class DropBlock2dOp final : public Operator<Context> {
 public:
  DropBlock2dOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        block_size_(OpArg<int64_t>("block_size", 7)),
        alpha_(OpArg<float>("alpha", 1.f)),
        decrement_(OpArg<float>("decrement", 0.f)) {
    GET_ARG_WITH_DESC(float, keep_prob, 0.9f);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t block_size_;
  float alpha_, decrement_, prob_ = 1.;
  DECLARE_ARG_WITH_DESC(float, keep_prob);
};

template <class Context>
class DropBlock2dGradientOp final : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(DropBlock2dGradientOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

DEFINE_ARG_WITH_DESC(float, DropBlock2dOp, keep_prob);

} // namespace dragon

#endif // DRAGON_OPERATORS_ACTIVATION_DROP_BLOCK_OP_H_
