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

#ifndef DRAGON_OPERATORS_MATH_CAST_OP_H_
#define DRAGON_OPERATORS_MATH_CAST_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class CastOp : public Operator<Context> {
 public:
  SIMPLE_CTOR_DTOR(CastOp);
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

  template <typename InputT, typename OutputT>
  bool MaybeConvert();
};

template <class Context>
class CastGradientOp final : public CastOp<Context> {
 public:
  CastGradientOp(const OperatorDef& def, Workspace* ws)
      : CastOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_CAST_OP_H_
