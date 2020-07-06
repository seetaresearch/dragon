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

#ifndef DRAGON_OPERATORS_MATH_FULLY_CONNECTED_OP_H_
#define DRAGON_OPERATORS_MATH_FULLY_CONNECTED_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class FullyConnectedOp final : public Operator<Context> {
 public:
  FullyConnectedOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_channels_(OpArg<int64_t>("out_channels", 0)),
        transW_(OpArg<int64_t>("transW", 1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t out_channels_, transW_;
};

template <class Context>
class FullyConnectedGradientOp final : public Operator<Context> {
 public:
  FullyConnectedGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_channels_(OpArg<int64_t>("out_channels", 0)),
        transW_(OpArg<int64_t>("transW", 1)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t out_channels_, transW_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_FULLY_CONNECTED_OP_H_
