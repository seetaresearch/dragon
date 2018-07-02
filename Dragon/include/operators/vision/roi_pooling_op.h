// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_VISION_ROI_POOLING_OP_H_
#define DRAGON_OPERATORS_VISION_ROI_POOLING_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ROIPoolingOp final : public Operator<Context> {
 public:
    ROIPoolingOp(const OperatorDef& def, Workspace *ws) 
        : Operator<Context>(def, ws),
          pool_h(OperatorBase::Arg<int>("pool_h", 0)),
          pool_w(OperatorBase::Arg<int>("pool_w", 0)),
          spatial_scale(OperatorBase::Arg<float>("spatial_scale", 1.0)) {
        CHECK_GT(pool_h, 0) << "\npool_h must > 0";
        CHECK_GT(pool_w, 0) << "\npool_w must > 0";
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int pool_h, pool_w;
    float spatial_scale;
};

template <class Context>
class ROIPoolingGradientOp final : public Operator<Context> {
 public:
    ROIPoolingGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          pool_h(OperatorBase::Arg<int>("pool_h", 0)),
          pool_w(OperatorBase::Arg<int>("pool_w", 0)),
          spatial_scale(OperatorBase::Arg<float>("spatial_scale", 1.f)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int pool_h, pool_w;
    float spatial_scale;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_VISION_ROI_POOLING_OP_H_