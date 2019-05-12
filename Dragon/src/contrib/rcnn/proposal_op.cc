#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "contrib/rcnn/proposal_op.h"
#include "contrib/rcnn/bbox_utils.h"

namespace dragon {

template <class Context> template <typename T>
void ProposalOp<Context>::RCNNImpl() {
    using BT = float;  // DType of BBox
    using BC = CPUContext;  // Context of BBox

    int feat_h, feat_w, K, A;
    int total_rois = 0, num_rois;
    int num_candidates, num_proposals;

    auto* batch_scores = X(-3).template data<T, BC>();
    auto* batch_deltas = X(-2).template data<T, BC>();
    auto* im_info = X(-1).template data<BT, BC>();
    auto* y = Y(0)->template mutable_data<BT, BC>();

    for (int n = 0; n < num_images_; ++n) {
        const BT im_h = im_info[0];
        const BT im_w = im_info[1];
        const BT scale = im_info[2];
        const BT min_box_h = min_size_ * scale;
        const BT min_box_w = min_size_ * scale;
        auto* scores = batch_scores + n * X(-3).stride(0);
        auto* deltas = batch_deltas + n * X(-2).stride(0);
        if (strides_.size() == 1) {
            // Case 1: single stride
            feat_h = X(0).dim(2);
            feat_w = X(0).dim(3);
            K = feat_h * feat_w;
            A = int(ratios_.size() * scales_.size());
            // Select the Top-K candidates as proposals
            num_candidates = A * K;
            num_proposals = std::min(
                num_candidates,
                (int)pre_nms_topn_
            );
            utils::math::ArgPartition(
                num_candidates,
                num_proposals,
                true, scores, indices_
            );
            // Decode the candidates
            anchors_.resize((size_t)(A * 4));
            proposals_.Reshape({ num_proposals, 5 });
            rcnn::GenerateAnchors(
                strides_[0],
                (int)ratios_.size(),
                (int)scales_.size(),
                ratios_.data(),
                scales_.data(),
                anchors_.data()
            );
            rcnn::GenerateGridAnchors(
                num_proposals, A,
                feat_h, feat_w,
                strides_[0],
                0,
                anchors_.data(),
                indices_.data(),
                proposals_.template mutable_data<BT, BC>()
            );
            rcnn::GenerateSSProposals(
                K, num_proposals,
                im_h, im_w,
                min_box_h, min_box_w,
                scores,
                deltas,
                indices_.data(),
                proposals_.template mutable_data<BT, BC>()
            );
            // Sort, NMS and Retrieve
            rcnn::SortProposals(
                0,
                num_proposals - 1,
                num_proposals,
                proposals_.template mutable_data<BT, BC>()
            );
            rcnn::ApplyNMS(
                num_proposals,
                post_nms_topn_,
                nms_thr_,
                proposals_.template mutable_data<BT, Context>(),
                roi_indices_.data(),
                num_rois, ctx()
            );
            rcnn::RetrieveRoIs(
                num_rois,
                n,
                proposals_.template data<BT, BC>(),
                roi_indices_.data(),
                y
            );
        } else if (strides_.size() > 1) {
            // Case 2: multiple stridess
            CHECK_EQ(strides_.size(), XSize() - 3)
                << "\nGiven " << strides_.size() << " strides "
                << "and " << XSize() - 3 << " feature inputs";
            CHECK_EQ(strides_.size(), scales_.size())
                << "\nGiven " << strides_.size() << " strides "
                << "and " << scales_.size() << " scales";
            // Select the Top-K candidates as proposals
            num_candidates = X(-3).dim(1);
            num_proposals = std::min(
                num_candidates,
                (int)pre_nms_topn_
            );
            utils::math::ArgPartition(
                num_candidates,
                num_proposals,
                true, scores, indices_
            );
            // Decode the candidates
            int base_offset = 0;
            proposals_.Reshape({ num_proposals, 5 });
            auto* proposals = proposals_
                .template mutable_data<BT, BC>();
            for (int i = 0; i < strides_.size(); i++) {
                feat_h = X(i).dim(2);
                feat_w = X(i).dim(3);
                K = feat_h * feat_w;
                A = (int)ratios_.size();
                anchors_.resize((size_t)(A * 4));
                rcnn::GenerateAnchors(
                    strides_[i],
                    (int)ratios_.size(),
                    1,
                    ratios_.data(),
                    scales_.data(),
                    anchors_.data()
                );
                rcnn::GenerateGridAnchors(
                    num_proposals, A,
                    feat_h, feat_w,
                    strides_[i],
                    base_offset,
                    anchors_.data(),
                    indices_.data(),
                    proposals
                );
                base_offset += (A * K);
            }
            rcnn::GenerateMSProposals(
                num_candidates,
                num_proposals,
                im_h, im_w,
                min_box_h, min_box_w,
                scores,
                deltas,
                &indices_[0],
                proposals
            );
            // Sort, NMS and Retrieve
            rcnn::SortProposals(
                0,
                num_proposals - 1,
                num_proposals,
                proposals
            );
            rcnn::ApplyNMS(
                num_proposals,
                post_nms_topn_,
                nms_thr_,
                proposals_.template mutable_data<BT, Context>(),
                roi_indices_.data(),
                num_rois, ctx()
            );
            rcnn::RetrieveRoIs(
                num_rois,
                n,
                proposals,
                roi_indices_.data(),
                y
            );
        } else {
            LOG(FATAL) << "Excepted at least one stride for proposals.";
        }
        total_rois += num_rois;
        y += (num_rois * 5);
        im_info += X(-1).dim(1);
    }

    Y(0)->Reshape({ total_rois, 5 });

    // Distribute rois into K bins
    if (YSize() > 1) {
        CHECK_EQ(max_level_ - min_level_ + 1, YSize())
            << "\nExcepted " << YSize() << " outputs for levels "
               "between [" << min_level_ << ", " << max_level_ << "].";
        vector<BT*> ys(YSize());
        vector<vec64_t> bins(YSize());
        Tensor RoIs; RoIs.ReshapeLike(*Y(0));

        auto* rois = RoIs.template mutable_data<BT, BC>();

        ctx()->template Copy<BT, BC, BC>(
            Y(0)->count(),
            rois, Y(0)->template data<BT, BC>()
        );

        rcnn::CollectRoIs(
            total_rois,
            min_level_,
            max_level_,
            canonical_level_,
            canonical_scale_,
            rois, bins
        );

        for (int i = 0; i < YSize(); i++) {
            Y(i)->Reshape({ std::max((int)bins[i].size(), 1), 5 });
            ys[i] = Y(i)->template mutable_data<BT, BC>();
        }

        rcnn::DistributeRoIs(bins, rois, ys);
    }
}

template <class Context> template <typename T>
void ProposalOp<Context>::RetinaNetImpl() {
    using BT = float;  // DType of BBox
    using BC = CPUContext;  // Context of BBox

    int feat_h, feat_w, C = X(-3).dim(2), A, K;
    int total_proposals = 0;
    int num_candidates, num_boxes, num_proposals;

    auto* batch_scores = X(-3).template data<T, BC>();
    auto* batch_deltas = X(-2).template data<T, BC>();
    auto* im_info = X(-1).template data<BT, BC>();
    auto* y = Y(0)->template mutable_data<BT, BC>();

    for (int n = 0; n < num_images_; ++n) {
        const BT im_h = im_info[0];
        const BT im_w = im_info[1];
        const BT im_scale = im_info[2];
        auto* scores = batch_scores + n * X(-3).stride(0);
        auto* deltas = batch_deltas + n * X(-2).stride(0);
        CHECK_EQ(strides_.size(), XSize() - 3)
            << "\nGiven " << strides_.size() << " strides "
            << "and " << XSize() - 3 << " features";
        // Select the Top-K candidates as proposals
        num_boxes = X(-3).dim(1);
        num_candidates = X(-3).count(1);
        roi_indices_.resize(num_candidates);
        num_candidates = 0;
        for (int i = 0; i < roi_indices_.size(); ++i)
            if (scores[i] > score_thr_)
                roi_indices_[num_candidates++] = i;
        scores_.resize(num_candidates);
        for (int i = 0; i < num_candidates; ++i)
            scores_[i] = scores[roi_indices_[i]];
        num_proposals = std::min(
            num_candidates,
            (int)pre_nms_topn_
        );
        utils::math::ArgPartition(
            num_candidates,
            num_proposals,
            true,
            scores_.data(),
            indices_
        );
        for (int i = 0; i < num_proposals; ++i)
            indices_[i] = roi_indices_[indices_[i]];
        // Decode the candidates
        int base_offset = 0;
        for (int i = 0; i < strides_.size(); i++) {
            feat_h = X(i).dim(2);
            feat_w = X(i).dim(3);
            K = feat_h * feat_w;
            A = int(ratios_.size() * scales_.size());
            anchors_.resize((size_t)(A * 4));
            rcnn::GenerateAnchors(
                strides_[i],
                (int)ratios_.size(),
                (int)scales_.size(),
                ratios_.data(),
                scales_.data(),
                anchors_.data()
            );
            rcnn::GenerateGridAnchors(
                num_proposals, C, A,
                feat_h, feat_w,
                strides_[i],
                base_offset,
                anchors_.data(),
                indices_.data(),
                y
            );
            base_offset += (A * K);
        }
        rcnn::GenerateMCProposals(
            num_proposals,
            num_boxes, C,
            n,
            im_h, im_w,
            im_scale,
            scores,
            deltas,
            indices_.data(),
            y
        );
        total_proposals += num_proposals;
        y += (num_proposals * 7);
        im_info += X(-1).dim(1);
    }

    Y(0)->Reshape({ total_proposals, 7 });
}

template <class Context>
void ProposalOp<Context>::RunOnDevice() {
    num_images_ = X(0).dim(0);

    CHECK_EQ(X(-1).dim(0), num_images_)
        << "\nExcepted " << num_images_
        << " groups info, while got "
        << X(-1).dim(0) << ".";

    if (det_type_ == "RCNN") {
        roi_indices_.resize(post_nms_topn_);
        Y(0)->Reshape({ num_images_ * post_nms_topn_, 5 });
        if (XIsType(X(-3), float)) {
            RCNNImpl<float>(); 
        } else { 
            LOG(FATAL) << DTypeString(
                X(0), { "float32" }
            ); 
        }
    } else if (det_type_ == "RETINANET") {
        Y(0)->Reshape({ num_images_ * pre_nms_topn_, 7 });
        if (XIsType(X(-3), float)) {
            RetinaNetImpl<float>();
        } else {
            LOG(FATAL) << DTypeString(
                X(0), { "float32" }
            );
        }
    } else {
        LOG(FATAL) << "Unknown Detector: " << det_type_;
    }
}

DEPLOY_CPU(Proposal);
#ifdef WITH_CUDA
DEPLOY_CUDA(Proposal);
#endif

OPERATOR_SCHEMA(Proposal)
    .NumInputs(3, INT_MAX)
    .NumOutputs(1, INT_MAX);

}  // namespace dragon