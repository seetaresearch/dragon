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

#ifndef DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_
#define DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ProposalOp final : public Operator<Context> {
 public:
    ProposalOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          det_type(OperatorBase::Arg<string>("det_type", "RCNN")),
          strides(OperatorBase::Args<int64_t>("strides")),
          ratios(OperatorBase::Args<float>("ratios")),
          scales(OperatorBase::Args<float>("scales")),
          pre_nms_top_n(OperatorBase::Arg<int64_t>("pre_nms_top_n", 6000)),
          post_nms_top_n(OperatorBase::Arg<int64_t>("post_nms_top_n", 300)),
          nms_thresh(OperatorBase::Arg<float>("nms_thresh", 0.7f)),
          score_thresh(OperatorBase::Arg<float>("score_thresh", 0.05f)),
          min_size(OperatorBase::Arg<int64_t>("min_size", 16)),
          min_level(OperatorBase::Arg<int64_t>("min_level", 2)),
          max_level(OperatorBase::Arg<int64_t>("max_level", 5)),
          canonical_level(OperatorBase::Arg<int64_t>("canonical_level", 4)),
          canonical_scale(OperatorBase::Arg<int64_t>("canonical_scale", 224)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

    template <typename T> void RunWithRCNN();
    template <typename T> void RunWithRetinaNet();

 protected:
    string det_type;
    float nms_thresh, score_thresh;
    vector<int64_t> strides, indices, roi_indices;
    vector<float> ratios, scales, scores_ex;
    int64_t pre_nms_top_n, post_nms_top_n, min_size, num_images;
    int64_t min_level, max_level, canonical_level, canonical_scale;
    Tensor anchors_, proposals_, nms_mask_;
};

}  // namespace dragon

#endif  // DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_