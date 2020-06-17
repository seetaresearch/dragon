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

#ifndef DRAGON_OPERATORS_VISION_DEPTH_TO_SPACE_OP_H_
#define DRAGON_OPERATORS_VISION_DEPTH_TO_SPACE_OP_H_

#include "dragon/operators/array/transpose_op.h"

namespace dragon {

template <class Context>
class DepthToSpaceOp final : public Operator<Context> {
 public:
  DepthToSpaceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), block_size_(OpArg<int>("block_size", 2)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t block_size_;
};

template <class Context>
class DepthToSpaceGradientOp final : public TransposeGradientOp<Context> {
 public:
  DepthToSpaceGradientOp(const OperatorDef& def, Workspace* ws)
      : TransposeGradientOp<Context>(def, ws) {}
};

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_DEPTH_TO_SPACE_OP_H_
