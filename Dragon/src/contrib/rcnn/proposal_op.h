// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_
#define DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ProposalOp final : public Operator<Context> {
 public:
    ProposalOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          strides(OperatorBase::GetRepeatedArg<int>("strides")),
          ratios(OperatorBase::GetRepeatedArg<float>("ratios")),
          scales(OperatorBase::GetRepeatedArg<float>("scales")),
          pre_nms_top_n(OperatorBase::GetSingleArg<int>("pre_nms_top_n", 6000)),
          post_nms_top_n(OperatorBase::GetSingleArg<int>("post_nms_top_n", 300)),
          nms_thresh(OperatorBase::GetSingleArg<float>("nms_thresh", (float)0.7)),
          min_size(OperatorBase::GetSingleArg<int>("min_size", 16)), 
          min_level(OperatorBase::GetSingleArg<int>("min_level", 2)),
          max_level(OperatorBase::GetSingleArg<int>("max_level", 5)),
          canonical_level(OperatorBase::GetSingleArg<int>("canonical_level", 4)),
          canonical_scale(OperatorBase::GetSingleArg<int>("canonical_scale", 224)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<int> strides;
    vector<float> ratios, scales;
    TIndex pre_nms_top_n, post_nms_top_n, min_size, num_images;
    TIndex min_level, max_level, canonical_level, canonical_scale;
    float nms_thresh;
    Tensor anchors_, proposals_, roi_indices_, nms_mask_;
};

}    // namespace dragon

#endif    // DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_