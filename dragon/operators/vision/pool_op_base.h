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

#ifndef DRAGON_OPERATORS_VISION_POOL_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_POOL_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class PoolOpBase : public Operator<Context> {
 public:
  PoolOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(OpArg<string>("mode", "MAX")),
        padding_(OpArg<string>("padding", "VALID")),
        ceil_mode_(OpArg<int64_t>("ceil_mode", 0)),
        global_pool_(OpArg<int64_t>("global_pooling", 0)) {
    if (data_format() == "NCHW")
      axis_ = 2;
    else if (data_format() == "NHWC")
      axis_ = 1;
    else
      LOG(FATAL) << "Unknown DataFormat: " << data_format();
    num_axes_ = -1; // Unknown
  }
  USE_OPERATOR_FUNCTIONS;

  void Setup(int num_axes);
  void ComputeOutShape();

 protected:
  string mode_, padding_;
  int64_t axis_, num_axes_;
  int64_t global_pool_, ceil_mode_;
  vec64_t kshape_, stride_, pad_l_, pad_r_;
  vec64_t in_dims_, out_dims_, out_shape_;
};

#define USE_POOLING_FUNCTIONS                 \
  using PoolOpBase<Context>::Setup;           \
  using PoolOpBase<Context>::ComputeOutShape; \
  using PoolOpBase<Context>::mode_;           \
  using PoolOpBase<Context>::kshape_;         \
  using PoolOpBase<Context>::stride_;         \
  using PoolOpBase<Context>::pad_l_;          \
  using PoolOpBase<Context>::pad_r_;          \
  using PoolOpBase<Context>::axis_;           \
  using PoolOpBase<Context>::num_axes_;       \
  using PoolOpBase<Context>::in_dims_;        \
  using PoolOpBase<Context>::out_dims_;       \
  using PoolOpBase<Context>::out_shape_;

} // namespace dragon

#endif // RAGON_OPERATORS_VISION_POOL_OP_BASE_H_
