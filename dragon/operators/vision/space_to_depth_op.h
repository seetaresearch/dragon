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

#ifndef DRAGON_OPERATORS_VISION_SPACE_TO_DEPTH_OP_H_
#define DRAGON_OPERATORS_VISION_SPACE_TO_DEPTH_OP_H_

#include "dragon/operators/array/transpose_op.h"

namespace dragon {

template <class Context>
class SpaceToDepthOp final : public Operator<Context> {
 public:
  SpaceToDepthOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), block_size_(OpArg<int>("block_size", 2)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t block_size_;
  Tensor X_, *X_strides_, *Y_dims_;
};

template <class Context>
class SpaceToDepthGradientOp final : public TransposeGradientOp<Context> {
 public:
  SpaceToDepthGradientOp(const OperatorDef& def, Workspace* ws)
      : TransposeGradientOp<Context>(def, ws) {}
};

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_SPACE_TO_DEPTH_OP_H_
