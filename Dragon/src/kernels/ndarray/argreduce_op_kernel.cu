#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Argmax <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Argmax(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 neg_bound,
    const T*                x,
    int64_t*                indices) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int max_idx = -1; T max_val = neg_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[idx] = max_idx;
    }
}

template <typename T>
__global__ void _Argmax_v2(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 neg_bound,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int max_idx = -1; T max_val = neg_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[idx] = max_idx;
        values[idx] = max_val;
    }
}

template<> void Argmax<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values,
    CUDAContext*            ctx) {
    CHECK_EQ(top_k, 1) << "top_k > 1 is not supported with CUDA";
    if (values == nullptr) {
        _Argmax<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, axis_dim, inner_dim,
                -FLT_MAX, x, indices);
    } else {
        _Argmax_v2<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, axis_dim, inner_dim,
                -FLT_MAX, x, indices, values);
    }
}

/*! Argmin <T = float32, Device = CUDA> */

template <typename T>
__global__ void _Argmin(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 pos_bound,
    const T*                x,
    int64_t*                indices) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int min_idx = -1; T min_val = pos_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        indices[idx] = min_idx;
    }
}

template <typename T>
__global__ void _Argmin_v2(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const T                 pos_bound,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int oix = idx / inner_dim;
        const int iix = idx % inner_dim;
        int min_idx = -1; T min_val = pos_bound;
        for (int j = 0; j < axis_dim; ++j) {
            const T val = x[(oix * axis_dim + j)
                                 * inner_dim + iix];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        indices[idx] = min_idx;
        values[idx] = min_val;
    }
}

template<> void Argmin<float, CUDAContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values,
    CUDAContext*            ctx) {
    CHECK_EQ(top_k, 1) << "top_k > 1 is not supported with CUDA";
    if (values == nullptr) {
        _Argmin<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, axis_dim, inner_dim,
                FLT_MAX, x, indices);
    } else {
        _Argmin_v2<float>
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >
            (count, axis_dim, inner_dim,
                FLT_MAX, x, indices, values);
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA