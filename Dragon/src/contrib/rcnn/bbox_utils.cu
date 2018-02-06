#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "contrib/rcnn/bbox_utils.h"

namespace dragon {

namespace rcnn {

/******************** BBox ********************/

template <typename T>
__device__ int _BBoxTransform(const T dx, const T dy,
                              const T d_log_w, const T d_log_h,
                              const T im_w, const T im_h,
                              const T min_box_w, const T min_box_h,
                              T* bbox) {
    const T w = bbox[2] - bbox[0] + (T)1;
    const T h = bbox[3] - bbox[1] + (T)1;
    const T ctr_x = bbox[0] + (T)0.5 * w;
    const T ctr_y = bbox[1] + (T)0.5 * h;

    const T pred_ctr_x = dx * w + ctr_x;
    const T pred_ctr_y = dy * h + ctr_y;
    const T pred_w = exp(d_log_w) * w;
    const T pred_h = exp(d_log_h) * h;

    bbox[0] = pred_ctr_x - (T)0.5 * pred_w;
    bbox[1] = pred_ctr_y - (T)0.5 * pred_h;
    bbox[2] = pred_ctr_x + (T)0.5 * pred_w;
    bbox[3] = pred_ctr_y + (T)0.5 * pred_h;

    bbox[0] = max((T)0, min(bbox[0], im_w - (T)1));
    bbox[1] = max((T)0, min(bbox[1], im_h - (T)1));
    bbox[2] = max((T)0, min(bbox[2], im_w - (T)1));
    bbox[3] = max((T)0, min(bbox[3], im_h - (T)1));

    const T box_w = bbox[2] - bbox[0] + (T)1;
    const T box_h = bbox[3] - bbox[1] + (T)1;
    return (box_w >= min_box_w) * (box_h >= min_box_h);
}

/******************** Proposal ********************/

template <typename T>
__global__ void _GenerateProposals(const int nthreads,
                                   const int A,
                                   const int feat_h,
                                   const int feat_w,
                                   const int stride,
                                   const float im_h, const float im_w,
                                   const float min_box_h, const float min_box_w,
                                   const T* scores,
                                   const T* bbox_deltas,
                                   const T* anchors,
                                   T* proposals) {
    CUDA_KERNEL_LOOP(idx, nthreads) {
        const int h = idx / A / feat_w;
        const int w = (idx / A) % feat_w;
        const int a = idx % A;
        const T x = w * stride;
        const T y = h * stride;
        const T* bbox_delta = bbox_deltas + h * feat_w + w;
        const T* score = scores + h * feat_w + w;
        const int K = feat_h * feat_w;
        const T dx = bbox_delta[(a * 4 + 0) * K];
        const T dy = bbox_delta[(a * 4 + 1) * K];
        const T d_log_w = bbox_delta[(a * 4 + 2) * K];
        const T d_log_h = bbox_delta[(a * 4 + 3) * K];
        T* proposal = proposals + idx * 5;
        proposal[0] = x + anchors[a * 4 + 0];
        proposal[1] = y + anchors[a * 4 + 1];
        proposal[2] = x + anchors[a * 4 + 2];
        proposal[3] = y + anchors[a * 4 + 3];
        proposal[4] = _BBoxTransform(dx, dy,
                           d_log_w, d_log_h,
                                 im_w, im_h,
                       min_box_w, min_box_h,
                   proposal) * score[a * K];
    }
}

template <> void GenerateProposals<float, CUDAContext>(const int A,
                                                       const int feat_h,
                                                       const int feat_w,
                                                       const int stride,
                                                       const float im_h, const float im_w,
                                                       const float min_box_h, const float min_box_w,
                                                       const float* scores,
                                                       const float* bbox_deltas,
                                                       const float* anchors,
                                                       float* proposals) {
    const int num_proposals = A * feat_h * feat_w;
    _GenerateProposals<float> << <GET_BLOCKS(num_proposals), CUDA_NUM_THREADS >> >(num_proposals,
                                                                                               A,
                                                                                          feat_h,
                                                                                          feat_w,
                                                                                          stride,
                                                                                      im_h, im_w,
                                                                            min_box_h, min_box_w,
                                                                                          scores,
                                                                                     bbox_deltas,
                                                                                         anchors,
                                                                                      proposals);
    CUDA_POST_KERNEL_CHECK;
}

template <typename T>
__global__ void _GenerateProposals_v2(const int nthreads,
                                      const float im_h, const float im_w,
                                      const float min_box_h, const float min_box_w,
                                      const T* scores,
                                      const T* bbox_deltas,
                                      T* proposals) {
    CUDA_KERNEL_LOOP(idx, nthreads) {
        const float dx = bbox_deltas[idx];
        const float dy = bbox_deltas[nthreads + idx];
        const float d_log_w = bbox_deltas[2 * nthreads + idx];
        const float d_log_h = bbox_deltas[3 * nthreads + idx];
        T* proposal = proposals + idx * 5;
        proposal[4] = _BBoxTransform(dx, dy,
                           d_log_w, d_log_h,
                                 im_w, im_h,
                       min_box_w, min_box_h,
                    proposal) * scores[idx];
    }
}

template <> void GenerateProposals_v2<float, CUDAContext>(const int total_anchors,
                                                          const float im_h, const float im_w,
                                                          const float min_box_h, const float min_box_w,
                                                          const float* scores,
                                                          const float* bbox_deltas,
                                                          float* proposals) {
    _GenerateProposals_v2<float> << <GET_BLOCKS(total_anchors), CUDA_NUM_THREADS >> >(total_anchors,
                                                                                         im_h, im_w,
                                                                               min_box_h, min_box_w,
                                                                                             scores,
                                                                                        bbox_deltas,
                                                                                         proposals);
    CUDA_POST_KERNEL_CHECK;
}

/******************** NMS ********************/

#define DIV_THEN_CEIL(x, y) (((x) + (y) - 1) / (y))
#define NMS_BLOCK_SIZE 64

template <typename T>
__device__  T iou(const T* A, const T* B) {
    const T x1 = max(A[0], B[0]);
    const T y1 = max(A[1], B[1]);
    const T x2 = min(A[2], B[2]);
    const T y2 = min(A[3], B[3]);
    const T width = max((T)0, x2 - x1 + (T)1);
    const T height = max((T)0, y2 - y1 + (T)1);
    const T area = width * height;
    const T A_area = (A[2] - A[0] + (T)1) * (A[3] - A[1] + (T)1);
    const T B_area = (B[2] - B[0] + (T)1) * (B[3] - B[1] + (T)1);
    return area / (A_area + B_area - area);
}

template <typename T>
__global__ static void nms_mask(const T boxes[],
                                unsigned long long mask[],
                                const int num_boxes,
                                const T nms_thresh) {
    const int i_start = blockIdx.x * NMS_BLOCK_SIZE;
    const int di_end = min(num_boxes - i_start, NMS_BLOCK_SIZE);
    const int j_start = blockIdx.y * NMS_BLOCK_SIZE;
    const int dj_end = min(num_boxes - j_start, NMS_BLOCK_SIZE);

    __shared__ T boxes_i[NMS_BLOCK_SIZE * 4];
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

    {
        const int dj = threadIdx.x;
        if (dj < dj_end) {
            const T* const box_j = boxes + (j_start + dj) * 5;
            unsigned long long mask_j = 0;
            const int di_start = (i_start == j_start) ? (dj + 1) : 0;
            for (int di = di_start; di < di_end; ++di) {
                const T* const box_i = boxes_i + di * 4;
                if (iou(box_j, box_i) > nms_thresh) {
                    mask_j |= 1ULL << di;
                }
            }
      {
          const int num_blocks = DIV_THEN_CEIL(num_boxes, NMS_BLOCK_SIZE);
          const int bi = blockIdx.x;
          mask[(j_start + dj) * num_blocks + bi] = mask_j;
      }
        }
    }
}

template <typename T>
void _NMS(const int num_boxes,
          const int max_keeps,
          const float thresh,
          const float* proposals,
          int* roi_indices,
          int& num_rois,
          Tensor* mask) {
    const int num_blocks = DIV_THEN_CEIL(num_boxes, NMS_BLOCK_SIZE);
    {
        const dim3 blocks(num_blocks, num_blocks);
        vector<TIndex> mask_shape(2);
        mask_shape[0] = num_boxes;
        mask_shape[1] = num_blocks * sizeof(unsigned long long) / sizeof(int);
        mask->Reshape(mask_shape);
        nms_mask << <blocks, NMS_BLOCK_SIZE >> >(
            proposals, (unsigned long long*)mask->template mutable_data<int, CUDAContext>(),
            num_boxes, thresh);
        CUDA_POST_KERNEL_CHECK;
    }
    // discard i-th box if it is significantly overlapped with
    // one or more previous (= scored higher) boxes
    {
        const unsigned long long* p_mask_cpu
            = (unsigned long long*)mask->mutable_data<int, CPUContext>();
        int num_selected = 0;
        vector<unsigned long long> dead_bit(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            dead_bit[i] = 0;
        }

        for (int i = 0; i < num_boxes; ++i) {
            const int nblock = i / NMS_BLOCK_SIZE;
            const int inblock = i % NMS_BLOCK_SIZE;

            if (!(dead_bit[nblock] & (1ULL << inblock))) {
                roi_indices[num_selected++] = i;
                const unsigned long long* const mask_i = p_mask_cpu + i * num_blocks;
                for (int j = nblock; j < num_blocks; ++j) {
                    dead_bit[j] |= mask_i[j];
                }

                if (num_selected == max_keeps) {
                    break;
                }
            }
        }
        num_rois = num_selected;
    }
}

template <> void NMS<float, CUDAContext>(const int num_boxes,
                                         const int max_keeps,
                                         const float thresh,
                                         const float* proposals,
                                         int* roi_indices,
                                         int& num_rois,
                                         Tensor* mask) {
    _NMS<float>(num_boxes, max_keeps, thresh,
                proposals, roi_indices, num_rois, mask);
}


}    // namespace rcnn

}    // namespace dragon

#endif // WITH_CUDA