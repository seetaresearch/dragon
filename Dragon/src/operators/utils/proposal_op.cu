#include "operators/utils/proposal_op.h"
#include "utils/cuda_device.h"

namespace dragon {

template <typename Dtype>
__device__ static int transform_box(Dtype box[], 
                                    const Dtype dx, 
                                    const Dtype dy,
                                    const Dtype d_log_w, 
                                    const Dtype d_log_h,
                                    const Dtype img_W, 
                                    const Dtype img_H,
                                    const Dtype min_box_W, 
                                    const Dtype min_box_H) {
    // width & height of box
    const Dtype w = box[2] - box[0] + (Dtype)1;
    const Dtype h = box[3] - box[1] + (Dtype)1;
    // center location of box
    const Dtype ctr_x = box[0] + (Dtype)0.5 * w;
    const Dtype ctr_y = box[1] + (Dtype)0.5 * h;

    // new center location according to gradient (dx, dy)
    const Dtype pred_ctr_x = dx * w + ctr_x;
    const Dtype pred_ctr_y = dy * h + ctr_y;
    // new width & height according to gradient d(log w), d(log h)
    const Dtype pred_w = exp(d_log_w) * w;
    const Dtype pred_h = exp(d_log_h) * h;

    // update upper-left corner location
    box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;
    box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;
    // update lower-right corner location
    box[2] = pred_ctr_x + (Dtype)0.5 * pred_w;
    box[3] = pred_ctr_y + (Dtype)0.5 * pred_h;

    // adjust new corner locations to be within the image region,
    box[0] = max((Dtype)0, min(box[0], img_W - (Dtype)1));
    box[1] = max((Dtype)0, min(box[1], img_H - (Dtype)1));
    box[2] = max((Dtype)0, min(box[2], img_W - (Dtype)1));
    box[3] = max((Dtype)0, min(box[3], img_H - (Dtype)1));

    // recompute new width & height
    const Dtype box_w = box[2] - box[0] + (Dtype)1;
    const Dtype box_h = box[3] - box[1] + (Dtype)1;

    // check if new box's size >= threshold
    return (box_w >= min_box_W) * (box_h >= min_box_H);
}

template <typename Dtype> 
static void sort_box(Dtype* list_cpu, const int start, const int end, const int num_top) {
    const Dtype pivot_score = list_cpu[start * 5 + 4];
    int left = start + 1, right = end;
    Dtype temp[5];
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

template <typename Dtype>
__global__ static void enumerate_proposals_gpu(const int nthreads,
                                               const Dtype bottom4d[],
                                               const Dtype d_anchor4d[],
                                               const Dtype anchors[],
                                               Dtype proposals[],
                                               const int num_anchors,
                                               const int bottom_H, const int bottom_W,
                                               const Dtype img_H, const Dtype img_W,
                                               const Dtype min_box_H, const Dtype min_box_W,
                                               const int feat_stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int h = index / num_anchors / bottom_W;
        const int w = (index / num_anchors) % bottom_W;
        const int k = index % num_anchors;
        const Dtype x = w * feat_stride;
        const Dtype y = h * feat_stride;
        const Dtype* p_box = d_anchor4d + h * bottom_W + w;
        const Dtype* p_score = bottom4d + h * bottom_W + w;

        const int bottom_area = bottom_H * bottom_W;
        const Dtype dx = p_box[(k * 4 + 0) * bottom_area];
        const Dtype dy = p_box[(k * 4 + 1) * bottom_area];
        const Dtype d_log_w = p_box[(k * 4 + 2) * bottom_area];
        const Dtype d_log_h = p_box[(k * 4 + 3) * bottom_area];

        Dtype* const p_proposal = proposals + index * 5;
        p_proposal[0] = x + anchors[k * 4 + 0];
        p_proposal[1] = y + anchors[k * 4 + 1];
        p_proposal[2] = x + anchors[k * 4 + 2];
        p_proposal[3] = y + anchors[k * 4 + 3];
        p_proposal[4]
            = transform_box(p_proposal,
            dx, dy, d_log_w, d_log_h,
            img_W, img_H, min_box_W, min_box_H)
            * p_score[k * bottom_area];
    }
}

template <typename Dtype>
__global__ static void retrieve_rois_gpu(const int nthreads,
                                         const int item_index,
                                         const Dtype proposals[],
                                         const int roi_indices[],
                                         Dtype rois[],
                                         Dtype roi_scores[]) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const Dtype* const proposals_index = proposals + roi_indices[index] * 5;
        rois[index * 5 + 0] = item_index;
        rois[index * 5 + 1] = proposals_index[0];
        rois[index * 5 + 2] = proposals_index[1];
        rois[index * 5 + 3] = proposals_index[2];
        rois[index * 5 + 4] = proposals_index[3];
        if (roi_scores) {
            roi_scores[index] = proposals_index[4];
        }
    }
}

template <typename Dtype>
__device__ static Dtype iou(const Dtype A[], const Dtype B[]) {
    // overlapped region (= box)
    const Dtype x1 = max(A[0], B[0]);
    const Dtype y1 = max(A[1], B[1]);
    const Dtype x2 = min(A[2], B[2]);
    const Dtype y2 = min(A[3], B[3]);

    // intersection area
    const Dtype width = max((Dtype)0, x2 - x1 + (Dtype)1);
    const Dtype height = max((Dtype)0, y2 - y1 + (Dtype)1);
    const Dtype area = width * height;

    // area of A, B
    const Dtype A_area = (A[2] - A[0] + (Dtype)1) * (A[3] - A[1] + (Dtype)1);
    const Dtype B_area = (B[2] - B[0] + (Dtype)1) * (B[3] - B[1] + (Dtype)1);

    // IoU
    return area / (A_area + B_area - area);
}

#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))

static const int nms_block_size = 64;

template <typename Dtype>
__global__ static void nms_mask(const Dtype boxes[], 
                                unsigned long long mask[], 
                                const int num_boxes, 
                                const Dtype nms_thresh) {
    // block region
    //   j = j_start + { 0, ..., dj_end - 1 }
    //   i = i_start + { 0, ..., di_end - 1 }
    const int i_start = blockIdx.x * nms_block_size;
    const int di_end = min(num_boxes - i_start, nms_block_size);
    const int j_start = blockIdx.y * nms_block_size;
    const int dj_end = min(num_boxes - j_start, nms_block_size);

    // copy all i-th boxes to GPU cache
    //   i = i_start + { 0, ..., di_end - 1 }
    __shared__ Dtype boxes_i[nms_block_size * 4];
    {
        const int di = threadIdx.x;
        if (di < di_end) {
            boxes_i[di * 4 + 0] = boxes[(i_start + di) * 5 + 0];
            boxes_i[di * 4 + 1] = boxes[(i_start + di) * 5 + 1];
            boxes_i[di * 4 + 2] = boxes[(i_start + di) * 5 + 2];
            boxes_i[di * 4 + 3] = boxes[(i_start + di) * 5 + 3];
        }
    }
    __syncthreads();

    // given j = j_start + dj,
    //   check whether box i is significantly overlapped with box j
    //   (i.e., IoU(box j, box i) > threshold)
    //   for all i = i_start + { 0, ..., di_end - 1 } except for i == j
    {
        const int dj = threadIdx.x;
        if (dj < dj_end) {
            // box j
            const Dtype* const box_j = boxes + (j_start + dj) * 5;

            // mask for significant overlap
            //   if IoU(box j, box i) > threshold,  di-th bit = 1
            unsigned long long mask_j = 0;

            // check for all i = i_start + { 0, ..., di_end - 1 }
            // except for i == j
            const int di_start = (i_start == j_start) ? (dj + 1) : 0;
            for (int di = di_start; di < di_end; ++di) {
                // box i
                const Dtype* const box_i = boxes_i + di * 4;

                // if IoU(box j, box i) > threshold,  di-th bit = 1
                if (iou(box_j, box_i) > nms_thresh) {
                    mask_j |= 1ULL << di;
                }
            }

            // mask: "num_boxes x num_blocks" array
            //   for mask[j][bi], "di-th bit = 1" means:
            //     box j is significantly overlapped with box i = i_start + di,
            //     where i_start = bi * block_size
      {
          const int num_blocks = DIV_THEN_CEIL(num_boxes, nms_block_size);
          const int bi = blockIdx.x;
          mask[(j_start + dj) * num_blocks + bi] = mask_j;
      }
        } // endif dj < dj_end
    }
}

template <typename Dtype>
void nms_gpu(const int num_boxes,
             const Dtype boxes_gpu[],
             Tensor* p_mask,
             int index_out_cpu[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh, 
             const int max_num_out) {

    const int num_blocks = DIV_THEN_CEIL(num_boxes, nms_block_size);

    {
        const dim3 blocks(num_blocks, num_blocks);
        vector<TIndex> mask_shape(2);
        mask_shape[0] = num_boxes;
        mask_shape[1] = num_blocks * sizeof(unsigned long long) / sizeof(int);
        p_mask->Reshape(mask_shape);

        // find all significantly-overlapped pairs of boxes
        nms_mask << <blocks, nms_block_size >> >(
            boxes_gpu, (unsigned long long*)p_mask->template mutable_data<int, CUDAContext>(),
            num_boxes, nms_thresh);
        CUDA_POST_KERNEL_CHECK;
    }

    // discard i-th box if it is significantly overlapped with
    // one or more previous (= scored higher) boxes
  {
      const unsigned long long* p_mask_cpu
          = (unsigned long long*)p_mask->mutable_data<int, CPUContext>();
      int num_selected = 0;
      vector<unsigned long long> dead_bit(num_blocks);
      for (int i = 0; i < num_blocks; ++i) {
          dead_bit[i] = 0;
      }

      for (int i = 0; i < num_boxes; ++i) {
          const int nblock = i / nms_block_size;
          const int inblock = i % nms_block_size;

          if (!(dead_bit[nblock] & (1ULL << inblock))) {
              index_out_cpu[num_selected++] = base_index + i;
              const unsigned long long* const mask_i = p_mask_cpu + i * num_blocks;
              for (int j = nblock; j < num_blocks; ++j) {
                  dead_bit[j] |= mask_i[j];
              }

              if (num_selected == max_num_out) {
                  break;
              }
          }
      }
      *num_out = num_selected;
  }
}

template
void nms_gpu(const int num_boxes, 
             const float boxes_gpu[], 
             Tensor* p_mask, 
             int index_out_cpu[],
             int* const num_out,
             const int base_index,
             const float nms_thresh, 
             const int max_num_out);

template
void nms_gpu(const int num_boxes,
             const double boxes_gpu[],
             Tensor* p_mask,
             int index_out_cpu[],
             int* const num_out,
             const int base_index,
             const double nms_thresh, 
             const int max_num_out);


template <class Context> template <typename T>
void ProposalOp<Context>::RunWithType() {
    auto* p_bottom_item = this->input(0).template data<T, CUDAContext>();
    auto* p_d_anchor_item = this->input(1).template data<T, CUDAContext>();
    auto* p_img_info_cpu = this->input(2).template data<T, CPUContext>();
    auto* p_roi_item = this->output(0)->template mutable_data<T, CUDAContext>();
    auto* p_score_item = (this->OutputSize() > 1) ? this->output(1)->template mutable_data<T, CUDAContext>() : NULL;

    vector<TIndex> proposals_shape(2), top_shape(2);
    proposals_shape[0] = 0; proposals_shape[1] = 5;
    top_shape[0] = 0; top_shape[1] = 5;

    for (int n = 0; n < this->input(0).dim(0); ++n) {
        // bottom shape: (2 x num_anchors) x H x W
        const int bottom_H = this->input(0).dim(2);
        const int bottom_W = this->input(0).dim(3);
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
        enumerate_proposals_gpu<T> << <GET_BLOCKS(num_proposals), CUDA_NUM_THREADS >> >(num_proposals,
                                                                        p_bottom_item + num_proposals, 
                                                                                      p_d_anchor_item,
                                                             anchors_.template data<T, CUDAContext>(), 
                                                   proposals_.template mutable_data<T, CUDAContext>(), 
                                                                                      anchors_.dim(0),
                                                                                   bottom_H, bottom_W, 
                                                                                         img_H, img_W, 
                                                                                 min_box_H, min_box_W,
                                                                                        feat_stride_);
        CUDA_POST_KERNEL_CHECK;
        sort_box<T>(proposals_.template mutable_data<T, CPUContext>(), 0, num_proposals - 1, pre_nms_topn_);
        nms_gpu<T>(pre_nms_topn, proposals_.template data<T, CUDAContext>(),
                                                                 &nms_mask_,
                      roi_indices_.template mutable_data<int, CPUContext>(),
                                                                  &num_rois,
                                                                          0, 
                                                                nms_thresh_, 
                                                            post_nms_topn_);

        retrieve_rois_gpu<T> << <GET_BLOCKS(num_rois), CUDA_NUM_THREADS >> >(num_rois, 
                                                                                    n, 
                                           proposals_.template data<T, CUDAContext>(),
                                       roi_indices_.template data<int, CUDAContext>(),
                                                                           p_roi_item, 
                                                                        p_score_item);
        CUDA_POST_KERNEL_CHECK;
        top_shape[0] += num_rois;
    }

    this->output(0)->Reshape(top_shape);
    if (this->OutputSize() > 1) {
        top_shape.pop_back();
        this->output(1)->Reshape(top_shape);
    }
}

template void ProposalOp<CUDAContext>::RunWithType<float>();

}