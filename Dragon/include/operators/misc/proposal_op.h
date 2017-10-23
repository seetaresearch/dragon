// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_MISC_PROPOSAL_OP_H_
#define DRAGON_OPERATORS_MISC_PROPOSAL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ProposalOp final : public Operator<Context> {
 public:
    ProposalOp(const OperatorDef& op_def, Workspace* ws) 
        : base_size_(OperatorBase::GetSingleArg<int>("base_size", 16)),
          min_size_(OperatorBase::GetSingleArg<int>("min_size", 16)),
          feat_stride_(OperatorBase::GetSingleArg<int>("feat_stride", -1)),
          pre_nms_topn_(OperatorBase::GetSingleArg<int>("pre_nms_topn", 12000)),
          post_nms_topn_(OperatorBase::GetSingleArg<int>("post_nms_topn", 2000)),
          nms_thresh_(OperatorBase::GetSingleArg<float>("nms_thresh", (float)0.7)),
          Operator<Context>(op_def, ws) { Setup(); }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int min_size_, base_size_, feat_stride_;
    int pre_nms_topn_, post_nms_topn_;
    float nms_thresh_;
    Tensor anchors_, roi_indices_, proposals_, nms_mask_;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_MISC_COMPARE_OP_H_