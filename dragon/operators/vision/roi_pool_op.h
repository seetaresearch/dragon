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

#ifndef DRAGON_OPERATORS_VISION_ROI_POOL_OP_H_
#define DRAGON_OPERATORS_VISION_ROI_POOL_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class RoiPoolOp final : public Operator<Context> {
 public:
  RoiPoolOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_h_(OP_SINGLE_ARG(int64_t, "pooled_h", 0)),
        out_w_(OP_SINGLE_ARG(int64_t, "pooled_w", 0)),
        spatial_scale_(OP_SINGLE_ARG(float, "spatial_scale", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float spatial_scale_;
  int64_t out_h_, out_w_;
};

template <class Context>
class RoiPoolGradientOp final : public Operator<Context> {
 public:
  RoiPoolGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_h_(OP_SINGLE_ARG(int64_t, "pooled_h", 0)),
        out_w_(OP_SINGLE_ARG(int64_t, "pooled_w", 0)),
        spatial_scale_(OP_SINGLE_ARG(float, "spatial_scale", 1.f)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  float spatial_scale_;
  int64_t out_h_, out_w_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_ROI_POOL_OP_H_
