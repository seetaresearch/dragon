#include "contrib/rcnn/proposal_op.h"
#include "contrib/rcnn/bbox_utils.h"

namespace dragon {

template <class Context> template <typename T>
void ProposalOp<Context>::RunWithType() {
    TIndex total_rois = 0;
    auto* im_info = Input(-1).template data<T, CPUContext>();
    auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
    for (int n = 0; n < num_images; ++n) {
        const T im_height = im_info[0];
        const T im_width = im_info[1];
        const T scale = im_info[2];
        const T min_box_h = min_size * scale;
        const T min_box_w = min_size * scale;
        int num_rois = 0;
        if (strides.size() == 1) {
            //  case 1: single stride (Faster R-CNN)
            const TIndex feat_height = Input(0).dim(2);
            const TIndex feat_width = Input(0).dim(3);
            const TIndex K = feat_height * feat_width;
            const TIndex A = ratios.size() * scales.size();
            const TIndex num_proposals = K * A;
            const TIndex pre_nms_topn = std::min(num_proposals, pre_nms_top_n);
            anchors_.Reshape({ A, 4 });
            proposals_.Reshape({ num_proposals, 5 });

            rcnn::GenerateAnchors<T>(strides[0],
                (int)ratios.size(), (int)scales.size(),
                    &ratios[0], &scales[0],
                        anchors_.template mutable_data<T, CPUContext>());

            rcnn::GenerateProposals<T, Context>(
                A, feat_height, feat_width, strides[0],
                    im_height, im_width, min_box_h, min_box_w,
                        Input(0).template data<T, Context>(),
                            Input(1).template data<T, Context>(),
                                anchors_.template mutable_data<T, Context>(),
                                    proposals_.template mutable_data<T, Context>());

            rcnn::SortProposals(0, num_proposals - 1, pre_nms_top_n,
                proposals_.template mutable_data<T, CPUContext>());

            rcnn::ApplyNMS<T, Context>(
                pre_nms_topn, post_nms_top_n, nms_thresh,
                    proposals_.template mutable_data<T, Context>(),
                        roi_indices_.template mutable_data<int, CPUContext>(), num_rois);

            rcnn::RetrieveRoIs<T>(num_rois, n,
                proposals_.template mutable_data<T, CPUContext>(),
                    roi_indices_.template mutable_data<int, CPUContext>(), Ydata);

        } else if (strides.size() > 1) {
            //  case 2: multiple stride (FPN / Mask R-CNN / RetinaNet)
            CHECK_EQ(strides.size(), (int)InputSize() - 3)
                << "\nGiven " << strides.size() << " strides and "
                << InputSize() - 3 << " feature inputs";
            CHECK_EQ(strides.size(), scales.size())
                << "\nGiven " << strides.size() << " strides and "
                << scales.size() << " scales";
            //  cls_probs: [1, total_proposals]
            //  bbox_deltas: [1, 4, total_proposals]
            TIndex total_proposals = Input(-3).dim(1), acc_proposals = 0;
            const TIndex pre_nms_topn = std::min(total_proposals, pre_nms_top_n);;
            proposals_.Reshape({ total_proposals, 5 });
            auto* proposals = proposals_.template mutable_data<T, CPUContext>();

            for (int i = 0; i < strides.size(); i++) {
                const TIndex feat_height = Input(i).dim(2);
                const TIndex feat_width = Input(i).dim(3);
                const TIndex K = feat_height * feat_width;
                const TIndex A = ratios.size();
                const TIndex num_proposals = K * A;
                anchors_.Reshape({ A, 4 });

                rcnn::GenerateAnchors<T>(strides[i],
                    (int)ratios.size(), 1, &ratios[0], &scales[0],
                        anchors_.template mutable_data<T, CPUContext>());

                rcnn::GenerateGridAnchors<T>(
                    A, feat_height, feat_width, strides[i],
                        anchors_.template mutable_data<T, CPUContext>(),
                            proposals);

                acc_proposals += num_proposals;
                proposals += (num_proposals * 5);
            }

            CHECK_EQ(acc_proposals, total_proposals)
                << "\nExcepted " << total_proposals << " proposals from the network, "
                << "but generated " << acc_proposals << " proposals.";

            rcnn::GenerateProposals_v2<T, Context>(total_proposals,
                im_height, im_width, min_box_h, min_box_w,
                    Input(-3).template data<T, Context>(),
                        Input(-2).template data<T, Context>(),
                            proposals_.template mutable_data<T, Context>());

            rcnn::SortProposals(0, total_proposals - 1, pre_nms_top_n,
                proposals_.template mutable_data<T, CPUContext>());

            rcnn::ApplyNMS<T, Context>(pre_nms_topn, post_nms_top_n, nms_thresh,
                proposals_.template mutable_data<T, Context>(),
                    roi_indices_.template mutable_data<int, CPUContext>(), num_rois);

            rcnn::RetrieveRoIs<T>(num_rois, n,
                proposals_.template mutable_data<T, CPUContext>(),
                    roi_indices_.template mutable_data<int, CPUContext>(), Ydata);

        } else {
            LOG(FATAL) << "There should be given at least one stride for proposals.";
        }
        total_rois += num_rois;
        Ydata += (num_rois * 5);
        im_info += Input(-1).dim(1);
    }
    Output(0)->Reshape(vector<TIndex>({ total_rois, 5 }));

    //  distribute rois into K bins
    if (OutputSize() > 1) {
        CHECK_EQ(max_level - min_level + 1, (int)OutputSize())
            << "\nExcepted " << OutputSize() << " outputs for levels between "
            << "[" << min_level << ", " << max_level << "].";
        vector< vector<TIndex> > roi_bins(OutputSize(), vector<TIndex>());
        vector<T*> outputs;
        Tensor collective_rois;
        collective_rois.ReshapeLike(*Output(0));
        auto* rois = collective_rois.template mutable_data<T, CPUContext>();

        CPUContext::template Copy<T, CPUContext, CPUContext>(
            collective_rois.count(), rois,
                Output(0)->template data<T, CPUContext>());

        rcnn::CollectRoIs<T>(total_rois,
            min_level, max_level,
                canonical_level, canonical_scale,
                    rois, roi_bins);

        for (int i = 0; i < OutputSize(); i++) {
            Output(i)->Reshape({ std::max((int)roi_bins[i].size(), 1), 5 });
            outputs.push_back(Output(i)->template mutable_data<T, CPUContext>());
        }
        rcnn::DistributeRoIs(roi_bins, rois, outputs);
    }
}

template <class Context>
void ProposalOp<Context>::RunOnDevice() {
    num_images = Input(0).dim(0);
    CHECK_EQ(Input(-1).dim(0), num_images)
        << "\nExcepted " << num_images << " groups image info, "
        << "but got " << Input(-1).dim(0) << ".";
    roi_indices_.Reshape({ post_nms_top_n });
    Output(0)->Reshape({ num_images * post_nms_top_n, 5 });

    if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
        if (Input(0).template IsType<float>()) RunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
        if (Input(0).template IsType<float>()) RunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
    }
}

DEPLOY_CPU(Proposal);
#ifdef WITH_CUDA
DEPLOY_CUDA(Proposal);
#endif

OPERATOR_SCHEMA(Proposal).NumInputs(3, INT_MAX).NumOutputs(1, INT_MAX);

}    // namespace dragon