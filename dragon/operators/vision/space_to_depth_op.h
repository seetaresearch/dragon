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

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class SpaceToDepthOp final : public Operator<Context> {
 public:
  SpaceToDepthOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        block_size_(OP_SINGLE_ARG(int, "block_size", 2)),
        mode_(OP_SINGLE_ARG(string, "mode", "DCR")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
  int64_t block_size_;
};

template <class Context>
class DepthToSpaceOp final : public Operator<Context> {
 public:
  DepthToSpaceOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        block_size_(OP_SINGLE_ARG(int, "block_size", 2)),
        mode_(OP_SINGLE_ARG(string, "mode", "DCR")) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Generic>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
  int64_t block_size_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_SPACE_TO_DEPTH_OP_H_
