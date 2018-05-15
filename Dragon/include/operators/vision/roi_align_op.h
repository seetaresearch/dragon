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

#ifndef DRAGON_OPERATORS_VISION_ROI_ALIGN_OP_H_
#define DRAGON_OPERATORS_VISION_ROI_ALIGN_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ROIAlignOp : public Operator<Context> {
 public:
    ROIAlignOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          pool_h(OperatorBase::GetSingleArg<int>("pool_h", 0)),
          pool_w(OperatorBase::GetSingleArg<int>("pool_w", 0)),
          spatial_scale(OperatorBase::GetSingleArg<float>("spatial_scale", 1.0)),
          sampling_ratio(OperatorBase::GetSingleArg<int>("sampling_ratio", 2)) {
        CHECK_GT(pool_h, 0) << "\npool_h must > 0";
        CHECK_GT(pool_w, 0) << "\npool_w must > 0";
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int pool_h, pool_w, sampling_ratio;
    float spatial_scale;
};

template <class Context>
class ROIAlignGradientOp : public Operator<Context> {
 public:
    ROIAlignGradientOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          pool_h(OperatorBase::GetSingleArg<int>("pool_h", 0)),
          pool_w(OperatorBase::GetSingleArg<int>("pool_w", 0)),
          spatial_scale(OperatorBase::GetSingleArg<float>("spatial_scale", 1.0)),
          sampling_ratio(OperatorBase::GetSingleArg<int>("sampling_ratio", 2)) {
        CHECK_GT(pool_h, 0) << "\npool_h must > 0";
        CHECK_GT(pool_w, 0) << "\npool_w must > 0";
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int pool_h, pool_w, sampling_ratio;
    float spatial_scale;
};

}    // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_ROI_ALIGN_OP_H_