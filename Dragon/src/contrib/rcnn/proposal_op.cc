#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "contrib/rcnn/proposal_op.h"
#include "contrib/rcnn/bbox_utils.h"

namespace dragon {

template <class Context> template <typename T>
void ProposalOp<Context>::RunWithType() {
    using BT = float;  // DType of BBox
    using BC = CPUContext;  // Context of BBox

    int feat_h, feat_w, K, A;
    int total_rois = 0, num_rois;
    int num_candidates, num_proposals;

    auto* RIdata = roi_indices.data();
    auto* batch_scores = Input(-3).template data<T, BC>();
    auto* batch_deltas = Input(-2).template data<T, BC>();
    auto* im_info = Input(-1).template data<BT, BC>();
    auto* Ydata = Output(0)->template mutable_data<BT, BC>();

    for (int n = 0; n < num_images; ++n) {
        const BT im_h = im_info[0];
        const BT im_w = im_info[1];
        const BT scale = im_info[2];
        const BT min_box_h = min_size * scale;
        const BT min_box_w = min_size * scale;
        auto* scores = batch_scores + n * Input(-3).stride(0);
        auto* deltas = batch_deltas + n * Input(-2).stride(0);
        if (strides.size() == 1) {
            // Case 1: single stride
            feat_h = Input(0).dim(2), feat_w = Input(0).dim(3);
            K = feat_h * feat_w, A = int(ratios.size() * scales.size());
            // Select the Top-K candidates as proposals
            num_candidates = K * A;
            num_proposals = std::min(
                num_candidates, (int)pre_nms_top_n);
            utils::math::ArgPartition(
                num_candidates, num_proposals,
                    true, scores, indices);
            // Decode the candidates
            anchors_.Reshape({ A, 4 });
            proposals_.Reshape({ num_proposals, 5 });
            auto* Adata = anchors_.template mutable_data<BT, BC>();
            auto* Pdata = proposals_.template mutable_data<BT, BC>();
            rcnn::GenerateAnchors(strides[0],
                (int)ratios.size(), (int)scales.size(),
                    &ratios[0], &scales[0], Adata);
            rcnn::GenerateGridAnchors(
                num_proposals, A, feat_h, feat_w,
                    strides[0], 0, Adata, indices.data(), Pdata);
            rcnn::GenerateSSProposals(K, num_proposals,
                im_h, im_w, min_box_h, min_box_w,
                    scores, deltas, indices.data(), Pdata);
            // Sort, NMS and Retrieve
            rcnn::SortProposals(0, num_proposals - 1, num_proposals, Pdata);
            rcnn::ApplyNMS(num_proposals, post_nms_top_n, nms_thresh,
                proposals_.template mutable_data<BT, Context>(),
                    RIdata, num_rois, ctx());
            rcnn::RetrieveRoIs(num_rois, n, Pdata, RIdata, Ydata);
        } else if (strides.size() > 1) {
            // Case 2: multiple stridess
            CHECK_EQ(strides.size(), InputSize() - 3)
                << "\nGiven " << strides.size() << " strides and "
                << InputSize() - 3 << " feature inputs";
            CHECK_EQ(strides.size(), scales.size())
                << "\nGiven " << strides.size() << " strides and "
                << scales.size() << " scales";
            // Select the Top-K candidates as proposals
            num_candidates = Input(-3).dim(1);
            num_proposals = std::min(
                num_candidates, (int)pre_nms_top_n);
            utils::math::ArgPartition(
                num_candidates, num_proposals,
                    true, scores, indices);
            // Decode the candidates
            int base_offset = 0;
            proposals_.Reshape({ num_proposals, 5 });
            auto* Pdata = proposals_.template mutable_data<BT, BC>();
            for (int i = 0; i < strides.size(); i++) {
                feat_h = Input(i).dim(2), feat_w = Input(i).dim(3);
                A = (int)ratios.size(), K = feat_h * feat_w;
                anchors_.Reshape({ A, 4 });
                auto* Adata = anchors_.template mutable_data<BT, BC>();
                rcnn::GenerateAnchors(
                    strides[i], (int)ratios.size(), 1,
                        &ratios[0], &scales[i], Adata);
                rcnn::GenerateGridAnchors(
                    num_proposals, A, feat_h, feat_w,
                        strides[i], base_offset,
                            Adata, indices.data(), Pdata);
                base_offset += K * A;
            }
            rcnn::GenerateMSProposals(
                num_candidates, num_proposals,
                    im_h, im_w, min_box_h, min_box_w,
                        scores, deltas, indices.data(), Pdata);
            // Sort, NMS and Retrieve
            rcnn::SortProposals(0, num_proposals - 1, num_proposals, Pdata);
            rcnn::ApplyNMS(num_proposals, post_nms_top_n, nms_thresh,
                proposals_.template mutable_data<BT, Context>(),
                    RIdata, num_rois, ctx());
            rcnn::RetrieveRoIs(num_rois, n, Pdata, RIdata, Ydata);
        } else {
            LOG(FATAL) << "Excepted at least one stride for proposals.";
        }
        total_rois += num_rois;
        Ydata += (num_rois * 5);
        im_info += Input(-1).dim(1);
    }

    Output(0)->Reshape({ total_rois, 5 });

    // Distribute rois into K bins
    if (OutputSize() > 1) {
        CHECK_EQ(max_level - min_level + 1, OutputSize())
            << "\nExcepted " << OutputSize() << " outputs for levels "
               "between [" << min_level << ", " << max_level << "].";
        vector<BT*> YSdata(OutputSize());
        vector< vector<int64_t> > bins(OutputSize());
        Tensor Y; Y.ReshapeLike(*Output(0));

        auto* rois = Y.template mutable_data<BT, BC>();
        ctx()->template Copy<BT, BC, BC>(Y.count(),
            rois, Output(0)->template data<BT, BC>());

        rcnn::CollectRoIs<BT>(total_rois, min_level, max_level,
            canonical_level, canonical_scale, rois, bins);

        for (int i = 0; i < OutputSize(); i++) {
            Output(i)->Reshape({ std::max((int)bins[i].size(), 1), 5 });
            YSdata[i] = Output(i)->template mutable_data<BT, BC>();
        }

        rcnn::DistributeRoIs(bins, rois, YSdata);
    }
}

template <class Context>
void ProposalOp<Context>::RunOnDevice() {
    num_images = Input(0).dim(0);
    CHECK_EQ(Input(-1).dim(0), num_images)
        << "\nExcepted " << num_images << " groups image info, "
        << "but got " << Input(-1).dim(0) << ".";
    roi_indices.resize(post_nms_top_n);
    Output(0)->Reshape({ num_images * post_nms_top_n, 5 });

    if (XIsType(Input(-3), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Proposal);
#ifdef WITH_CUDA
DEPLOY_CUDA(Proposal);
#endif

OPERATOR_SCHEMA(Proposal)
    .NumInputs(3, INT_MAX)
    .NumOutputs(1, INT_MAX);

}  // namespace dragon