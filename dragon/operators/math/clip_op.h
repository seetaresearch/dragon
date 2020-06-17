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

#ifndef DRAGON_OPERATORS_MATH_CLIP_OP_H_
#define DRAGON_OPERATORS_MATH_CLIP_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ClipOp : public Operator<Context> {
 public:
  ClipOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        low_(OpArg<float>("low", -FLT_MAX)),
        high_(OpArg<float>("high", FLT_MAX)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  pair<float, float> ComputeBoundsWithType();

  template <typename T>
  void DoRunWithType();

 protected:
  float low_, high_;
};

template <class Context>
class ClipGradientOp final : public ClipOp<Context> {
 public:
  ClipGradientOp(const OperatorDef& def, Workspace* ws)
      : ClipOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

} // namespace dragon

#endif // DRAGON_OPERATORS_MATH_CLIP_OP_H_
