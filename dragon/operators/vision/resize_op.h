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

#ifndef DRAGON_OPERATORS_VISION_RESIZE_OP_H_
#define DRAGON_OPERATORS_VISION_RESIZE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ResizeOp final : public Operator<Context> {
 public:
  ResizeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(str::upper(OP_SINGLE_ARG(string, "mode", "NEAREST"))),
        align_corners_(OP_SINGLE_ARG(int64_t, "align_corners", 0)) {
    INITIALIZE_OP_REPEATED_ARG(float, scales);
    INITIALIZE_OP_REPEATED_ARG(int64_t, sizes);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
  int64_t align_corners_;
  vec64_t in_dims_, out_dims_, out_shape_;
  DECLARE_OP_REPEATED_ARG(float, scales);
  DECLARE_OP_REPEATED_ARG(int64_t, sizes);
};

template <class Context>
class ResizeGradientOp final : public Operator<Context> {
 public:
  ResizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(str::upper(OP_SINGLE_ARG(string, "mode", "NEAREST"))),
        align_corners_(OP_SINGLE_ARG(int64_t, "align_corners", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  string mode_;
  int64_t align_corners_;
  vec64_t in_dims_, out_dims_;
};

DEFINE_OP_REPEATED_ARG(float, ResizeOp, scales);
DEFINE_OP_REPEATED_ARG(int64_t, ResizeOp, sizes);

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_RESIZE_OP_H_
