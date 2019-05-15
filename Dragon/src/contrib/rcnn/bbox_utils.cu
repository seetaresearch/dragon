#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "contrib/rcnn/bbox_utils.h"

namespace dragon {

namespace rcnn {

#define DIV_UP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define NMS_BLOCK_SIZE 64

template <typename T>
__device__  T _IoU(const T* A, const T* B) {
    const T x1 = max(A[0], B[0]);
    const T y1 = max(A[1], B[1]);
    const T x2 = min(A[2], B[2]);
    const T y2 = min(A[3], B[3]);
    const T width = max((T)0, x2 - x1 + 1);
    const T height = max((T)0, y2 - y1 + 1);
    const T area = width * height;
    const T A_area = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
    const T B_area = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
    return area / (A_area + B_area - area);
}

template <typename T>
__global__ void nms_mask(
    const int               num_boxes,
    const T                 nms_thresh,
    const T*                boxes,
    uint64_t*               mask) {
    const int i_start = blockIdx.x * NMS_BLOCK_SIZE;
    const int di_end = min(num_boxes - i_start, NMS_BLOCK_SIZE);
    const int j_start = blockIdx.y * NMS_BLOCK_SIZE;
    const int dj_end = min(num_boxes - j_start, NMS_BLOCK_SIZE);

    const int num_blocks = DIV_UP(num_boxes, NMS_BLOCK_SIZE);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ T boxes_i[NMS_BLOCK_SIZE * 4];

    if (tid < di_end) {
        boxes_i[tid * 4 + 0] = boxes[(i_start + tid) * 5 + 0];
        boxes_i[tid * 4 + 1] = boxes[(i_start + tid) * 5 + 1];
        boxes_i[tid * 4 + 2] = boxes[(i_start + tid) * 5 + 2];
        boxes_i[tid * 4 + 3] = boxes[(i_start + tid) * 5 + 3];
    }

    __syncthreads();

    if (tid < dj_end) {
        const T* const box_j = boxes + (j_start + tid) * 5;
        unsigned long long mask_j = 0;
        const int di_start = (i_start == j_start) ? (tid + 1) : 0;
        for (int di = di_start; di < di_end; ++di)
            if (_IoU(box_j, boxes_i + di * 4) > nms_thresh)
        mask_j |= 1ULL << di;
        mask[(j_start + tid) * num_blocks + bid] = mask_j;
    }
}

template <typename T>
void _ApplyNMS(
    const int               num_boxes,
    const int               max_keeps,
    const float             thresh,
    const T*                boxes,
    int64_t*                keep_indices,
    int&                    num_keep,
    CUDAContext*            ctx) {
    const int num_blocks = DIV_UP(num_boxes, NMS_BLOCK_SIZE);
    const dim3 blocks(num_blocks, num_blocks);
    size_t mask_nbytes = num_boxes * num_blocks * sizeof(uint64_t);
    size_t boxes_nbytes = num_boxes * 5 * sizeof(T);

    void* boxes_dev, *mask_dev;
    CUDA_CHECK(cudaMalloc(&boxes_dev, boxes_nbytes));
    CUDA_CHECK(cudaMalloc(&mask_dev, mask_nbytes));
    CUDA_CHECK(cudaMemcpy(boxes_dev, boxes,
        boxes_nbytes, cudaMemcpyHostToDevice));
    nms_mask<T>
        <<< blocks, NMS_BLOCK_SIZE,
            0, ctx->cuda_stream() >>> (num_boxes,
                 thresh, (T*)boxes_dev, (uint64_t*)mask_dev);
    ctx->FinishDeviceCompution();

    std::vector<uint64_t> mask_host(num_boxes * num_blocks);
    CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev,
        mask_nbytes, cudaMemcpyDeviceToHost));

    std::vector<uint64_t> dead_bit(num_blocks);
    memset(&dead_bit[0], 0, sizeof(uint64_t) * num_blocks);
    int num_selected = 0;

    for (int i = 0; i < num_boxes; ++i) {
        const int nblock = i / NMS_BLOCK_SIZE;
        const int inblock = i % NMS_BLOCK_SIZE;
        if (!(dead_bit[nblock] & (1ULL << inblock))) {
            keep_indices[num_selected++] = i;
            uint64_t* mask_i = &mask_host[0] + i * num_blocks;
            for (int j = nblock; j < num_blocks; ++j) dead_bit[j] |= mask_i[j];
            if (num_selected == max_keeps) break;
        }
    }
    num_keep = num_selected;
    CUDA_CHECK(cudaFree(mask_dev));
    CUDA_CHECK(cudaFree(boxes_dev));
}

template <> void ApplyNMS<float, CUDAContext>(
    const int               num_boxes,
    const int               max_keeps,
    const float             thresh,
    const float*            boxes,
    int64_t*                keep_indices,
    int&                    num_keep,
    CUDAContext*            ctx) {
    _ApplyNMS<float>(num_boxes, max_keeps, thresh,
        boxes, keep_indices, num_keep, ctx);
}

}  // namespace rcnn

}  // namespace dragon

#endif  // WITH_CUDA