#include "operators/utils/proposal_op.h"

namespace dragon {

#define ROUND(x) ((int)((x) + (T)0.5))

template <typename T>
static void generate_anchors(int base_size,
                      const T ratios[],
                      const T scales[],
                      const int num_ratios,
                      const int num_scales,
                      T anchors[]) {
  // base box's width & height & center location
  const T base_area = (T)(base_size * base_size);
  const T center = (T)0.5 * (base_size - (T)1);

  // enumerate all transformed boxes
  T* p_anchors = anchors;
  for (int i = 0; i < num_ratios; ++i) {
    // transformed width & height for given ratio factors
    const T ratio_w = (T)ROUND(sqrt(base_area / ratios[i]));
    const T ratio_h = (T)ROUND(ratio_w * ratios[i]);

    for (int j = 0; j < num_scales; ++j) {
      // transformed width & height for given scale factors
      const T scale_w = (T)0.5 * (ratio_w * scales[j] - (T)1);
      const T scale_h = (T)0.5 * (ratio_h * scales[j] - (T)1);

      // (x1, y1, x2, y2) for transformed box
      p_anchors[0] = center - scale_w;
      p_anchors[1] = center - scale_h;
      p_anchors[2] = center + scale_w;
      p_anchors[3] = center + scale_h;
      p_anchors += 4;
    } // endfor j
  }
}

template <typename T>
static int transform_box(T box[],
                         const T dx, const T dy,
                         const T d_log_w, const T d_log_h,
                         const T img_W, const T img_H,
                         const T min_box_W, const T min_box_H) {
    // width & height of box
    const T w = box[2] - box[0] + (T)1;
    const T h = box[3] - box[1] + (T)1;
    // center location of box
    const T ctr_x = box[0] + (T)0.5 * w;
    const T ctr_y = box[1] + (T)0.5 * h;

    // new center location according to gradient (dx, dy)
    const T pred_ctr_x = dx * w + ctr_x;
    const T pred_ctr_y = dy * h + ctr_y;
    // new width & height according to gradient d(log w), d(log h)
    const T pred_w = exp(d_log_w) * w;
    const T pred_h = exp(d_log_h) * h;

    // update upper-left corner location
    box[0] = pred_ctr_x - (T)0.5 * pred_w;
    box[1] = pred_ctr_y - (T)0.5 * pred_h;
    // update lower-right corner location
    box[2] = pred_ctr_x + (T)0.5 * pred_w;
    box[3] = pred_ctr_y + (T)0.5 * pred_h;

    // adjust new corner locations to be within the image region,
    box[0] = std::max((T)0, std::min(box[0], img_W - (T)1));
    box[1] = std::max((T)0, std::min(box[1], img_H - (T)1));
    box[2] = std::max((T)0, std::min(box[2], img_W - (T)1));
    box[3] = std::max((T)0, std::min(box[3], img_H - (T)1));

    // recompute new width & height
    const T box_w = box[2] - box[0] + (T)1;
    const T box_h = box[3] - box[1] + (T)1;

    // check if new box's size >= threshold
    return (box_w >= min_box_W) * (box_h >= min_box_H);
}

template <typename T>
static void enumerate_proposals_cpu(const T bottom4d[],
                                    const T d_anchor4d[],
                                    const T anchors[],
                                    T proposals[],
                                    const int num_anchors,
                                    const int bottom_H, const int bottom_W,
                                    const T img_H, const T img_W,
                                    const T min_box_H, const T min_box_W,
                                    const int feat_stride) {
    T* p_proposal = proposals;
    const int bottom_area = bottom_H * bottom_W;

    for (int h = 0; h < bottom_H; ++h) {
        for (int w = 0; w < bottom_W; ++w) {
            const T x = w * feat_stride;
            const T y = h * feat_stride;
            const T* p_box = d_anchor4d + h * bottom_W + w;
            const T* p_score = bottom4d + h * bottom_W + w;
            for (int k = 0; k < num_anchors; ++k) {
                const T dx = p_box[(k * 4 + 0) * bottom_area];
                const T dy = p_box[(k * 4 + 1) * bottom_area];
                const T d_log_w = p_box[(k * 4 + 2) * bottom_area];
                const T d_log_h = p_box[(k * 4 + 3) * bottom_area];
                p_proposal[0] = x + anchors[k * 4 + 0];
                p_proposal[1] = y + anchors[k * 4 + 1];
                p_proposal[2] = x + anchors[k * 4 + 2];
                p_proposal[3] = y + anchors[k * 4 + 3];
                p_proposal[4]
                    = transform_box(p_proposal,
                    dx, dy, d_log_w, d_log_h,
                    img_W, img_H, min_box_W, min_box_H)
                    * p_score[k * bottom_area];
                p_proposal += 5;
            } // endfor k
        } // endfor w
    } // endfor h
}

template <typename T>
static void sort_box(T list_cpu[], 
                     const int start, 
                     const int end, 
                     const int num_top) {
    const T pivot_score = list_cpu[start * 5 + 4];
    int left = start + 1, right = end;
    T temp[5];
    while (left <= right) {
        while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;
        while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;
        if (left <= right) {
            for (int i = 0; i < 5; ++i) {
                temp[i] = list_cpu[left * 5 + i];
            }
            for (int i = 0; i < 5; ++i) {
                list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
            }
            for (int i = 0; i < 5; ++i) {
                list_cpu[right * 5 + i] = temp[i];
            }
            ++left;
            --right;
        }
    }

    if (right > start) {
        for (int i = 0; i < 5; ++i) {
            temp[i] = list_cpu[start * 5 + i];
        }
        for (int i = 0; i < 5; ++i) {
            list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
        }
        for (int i = 0; i < 5; ++i) {
            list_cpu[right * 5 + i] = temp[i];
        }
    }

    if (start < right - 1) {
        sort_box(list_cpu, start, right - 1, num_top);
    }
    if (right + 1 < num_top && right + 1 < end) {
        sort_box(list_cpu, right + 1, end, num_top);
    }
}

template <typename T>
static T iou(const T A[], const T B[]) {
    if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) return 0;

    // overlapped region (= box)
    const T x1 = std::max(A[0], B[0]);
    const T y1 = std::max(A[1], B[1]);
    const T x2 = std::min(A[2], B[2]);
    const T y2 = std::min(A[3], B[3]);

    // intersection area
    const T width = std::max((T)0, x2 - x1 + (T)1);
    const T height = std::max((T)0, y2 - y1 + (T)1);
    const T area = width * height;

    // area of A, B
    const T A_area = (A[2] - A[0] + (T)1) * (A[3] - A[1] + (T)1);
    const T B_area = (B[2] - B[0] + (T)1) * (B[3] - B[1] + (T)1);

    // IoU
    return area / (A_area + B_area - area);
}

template <typename T>
void nms_cpu(const int num_boxes,
             const T boxes[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const T nms_thresh, 
             const int max_num_out) {
    int count = 0;
    std::vector<char> is_dead(num_boxes);
    for (int i = 0; i < num_boxes; ++i) is_dead[i] = 0;
    for (int i = 0; i < num_boxes; ++i) {
        if (is_dead[i]) continue;
        index_out[count++] = base_index + i;
        if (count == max_num_out) break;
        for (int j = i + 1; j < num_boxes; ++j) 
            if (!is_dead[j] && iou(&boxes[i * 5], &boxes[j * 5]) > nms_thresh) is_dead[j] = 1;
    }
    *num_out = count;
    is_dead.clear();
}

template <typename T>
static void retrieve_rois_cpu(const int num_rois,
                              const int item_index,
                              const T proposals[],
                              const int roi_indices[],
                              T rois[], T roi_scores[]) {
    for (int i = 0; i < num_rois; ++i) {
        const T* const proposals_index = proposals + roi_indices[i] * 5;
        rois[i * 5 + 0] = item_index;
        rois[i * 5 + 1] = proposals_index[0];
        rois[i * 5 + 2] = proposals_index[1];
        rois[i * 5 + 3] = proposals_index[2];
        rois[i * 5 + 4] = proposals_index[3];
        if (roi_scores) {
            roi_scores[i] = proposals_index[4];
        }
    }
}

template <class Context> template <typename T>
void ProposalOp<Context>::RunWithType() {
    auto* p_bottom_item = input(0).template data<T, CPUContext>();
    auto* p_d_anchor_item = input(1).template data<T, CPUContext>();
    auto* p_img_info_cpu = input(2).template data<T, CPUContext>();
    auto* p_roi_item = output(0)->template mutable_data<T, CPUContext>();
    auto* p_score_item = (OutputSize() > 1) ? output(1)->template mutable_data<T, CPUContext>() : NULL;

    vector<TIndex> proposals_shape(2), top_shape(2);
    proposals_shape[0] = 0; proposals_shape[1] = 5;
    top_shape[0] = 0; top_shape[1] = 5;

    for (int n = 0; n < input(0).dim(0); ++n) {
        // bottom shape: (2 x num_anchors) x H x W
        const int bottom_H = input(0).dim(2);
        const int bottom_W = input(0).dim(3);
        // input image height & width
        const T img_H = p_img_info_cpu[0];
        const T img_W = p_img_info_cpu[1];
        // scale factor for height & width
        const T scale_H = p_img_info_cpu[2];
        const T scale_W = p_img_info_cpu[3];
        // minimum box width & height
        const T min_box_H = min_size_ * scale_H;
        const T min_box_W = min_size_ * scale_W;
        // number of all proposals = num_anchors * H * W
        const int num_proposals = anchors_.dim(0) * bottom_H * bottom_W;
        // number of top-n proposals before NMS
        const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
        // number of final RoIs
        int num_rois = 0;

        // enumerate all proposals
        //   num_proposals = num_anchors * H * W
        //   (x1, y1, x2, y2, score) for each proposal
        // NOTE: for bottom, only foreground scores are passed
        proposals_shape[0] = num_proposals;
        proposals_.Reshape(proposals_shape);
        enumerate_proposals_cpu(p_bottom_item + num_proposals, p_d_anchor_item,
            anchors_.template data<T, CPUContext>(), proposals_.mutable_data<T, CPUContext>(),
            anchors_.dim(0), bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W, feat_stride_);

        sort_box(proposals_.mutable_data<T, CPUContext>(), 0, num_proposals - 1, pre_nms_topn_);

        nms_cpu(pre_nms_topn, proposals_.template data<T, CPUContext>(),
            roi_indices_.mutable_data<int, CPUContext>(), &num_rois,
            0, nms_thresh_, post_nms_topn_);

        retrieve_rois_cpu(
            num_rois, n, proposals_.template data<T, CPUContext>(),
            roi_indices_.template data<int, CPUContext>(), p_roi_item, p_score_item);

        top_shape[0] += num_rois;
    }

    output(0)->Reshape(top_shape);
    if (OutputSize() > 1) {
        top_shape.pop_back();
        output(1)->Reshape(top_shape);
    }
}

template <typename Context>
void ProposalOp<Context>::Setup() {
    vector<float> ratios(OperatorBase::GetRepeatedArg<float>("ratio"));
    vector<float> scales(OperatorBase::GetRepeatedArg<float>("scale"));
    vector<TIndex> anchors_shape(2);
    anchors_shape[0] = ratios.size() * scales.size();
    anchors_shape[1] = 4;
    anchors_.Reshape(anchors_shape);
    generate_anchors(base_size_, &ratios[0], &scales[0],
        (int)ratios.size(), (int)scales.size(),
        anchors_.mutable_data<float, CPUContext>());
    vector<TIndex> roi_indices_shape(1);
    roi_indices_shape[0] = post_nms_topn_;
    roi_indices_.Reshape(roi_indices_shape);

    // rois blob : holds R regions of interest, each is a 5 - tuple
    // (n, x1, y1, x2, y2) specifying an image batch index n and a
    // rectangle(x1, y1, x2, y2)
    vector<TIndex> top_shape(2);
    top_shape[0] = 1 * post_nms_topn_;
    top_shape[1] = 5;
    output(0)->Reshape(top_shape);

    // scores blob : holds scores for R regions of interest
    if (OutputSize() > 1) {
        top_shape.pop_back();
        output(0)->Reshape(top_shape);
    }
}
 
template <class Context>
void ProposalOp<Context>::RunOnDevice() {
    CHECK_EQ(input(0).dim(0), 1) << "only single item batches are supported.";

    if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
        if (input(0).template IsType<float>()) RunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
        if (input(0).template IsType<float>()) RunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";
    }
}

DEPLOY_CPU(Proposal);
#ifdef WITH_CUDA
DEPLOY_CUDA(Proposal);
#endif

OPERATOR_SCHEMA(Proposal).NumInputs(3).NumOutputs(1, 2);

}    // namespace dragon