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
          det_type_(OpArg<string>("det_type", "RCNN")),
          strides_(OpArgs<int64_t>("strides")),
          ratios_(OpArgs<float>("ratios")),
          scales_(OpArgs<float>("scales")),
          pre_nms_topn_(OpArg<int64_t>("pre_nms_top_n", 6000)),
          post_nms_topn_(OpArg<int64_t>("post_nms_top_n", 300)),
          nms_thr_(OpArg<float>("nms_thresh", 0.7f)),
          score_thr_(OpArg<float>("score_thresh", 0.05f)),
          min_size_(OpArg<int64_t>("min_size", 16)),
          min_level_(OpArg<int64_t>("min_level", 2)),
          max_level_(OpArg<int64_t>("max_level", 5)),
          canonical_level_(OpArg<int64_t>("canonical_level", 4)),
          canonical_scale_(OpArg<int64_t>("canonical_scale", 224)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

    template <typename T> void RCNNImpl();
    template <typename T> void RetinaNetImpl();

 protected:
    string det_type_;
    float nms_thr_, score_thr_;
    vec64_t strides_, indices_, roi_indices_;
    vector<float> ratios_, scales_, scores_, anchors_;
    int64_t min_size_, pre_nms_topn_, post_nms_topn_;
    int64_t num_images_, min_level_, max_level_;
    int64_t canonical_level_, canonical_scale_;
    Tensor proposals_, nms_mask_;
};

}  // namespace dragon

#endif  // DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_