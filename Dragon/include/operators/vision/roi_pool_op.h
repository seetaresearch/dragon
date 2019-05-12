/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_ROI_POOL_OP_H_
#define DRAGON_OPERATORS_VISION_ROI_POOL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ROIPoolOp final : public Operator<Context> {
 public:
    ROIPoolOp(const OperatorDef& def, Workspace *ws)
        : Operator<Context>(def, ws),
          pool_h_(OpArg<int64_t>("pool_h", 0)),
          pool_w_(OpArg<int64_t>("pool_w", 0)),
          spatial_scale_(OpArg<float>("spatial_scale", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float spatial_scale_;
    int64_t pool_h_, pool_w_;
};

template <class Context>
class ROIPoolGradientOp final : public Operator<Context> {
 public:
    ROIPoolGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          pool_h_(OpArg<int64_t>("pool_h", 0)),
          pool_w_(OpArg<int64_t>("pool_w", 0)),
          spatial_scale_(OpArg<float>("spatial_scale", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunImplFloat16();

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    float spatial_scale_;
    int64_t pool_h_, pool_w_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_ROI_POOL_OP_H_