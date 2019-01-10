#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Argmax <T = ?, Device = CUDA> */

template <typename T>
__global__ void _ArgMax(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int oix = y_idx / inner_dim;
        const int iix = y_idx % inner_dim;
        const T* X = x + (oix * axis_dim * inner_dim + iix);
        T max_val = X[0], val; int64_t max_idx = 0;
        for (int j = 1; j < axis_dim; ++j) {
            val = X[j * inner_dim];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[y_idx] = max_idx;
        if (values) values[y_idx] = max_val;
    }
}

/*! ArgMax <T = float16, Device = CUDA> */

__global__ void _ArgMaxHalf(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const half*             x,
    int64_t*                indices,
    half*                   values) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int oix = y_idx / inner_dim;
        const int iix = y_idx % inner_dim;
        const half* X = x + (oix * axis_dim * inner_dim + iix);
        half max_val = X[0], val; int64_t max_idx = 0;
        for (int j = 1; j < axis_dim; ++j) {
            val = X[j * inner_dim];
            if (__hgt(val, max_val)) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[y_idx] = max_idx;
        if (values) values[y_idx] = max_val;
#endif
    }
}

/*! ArgMin <T = ?, Device = CUDA> */

template <typename T>
__global__ void _ArgMin(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        const int oix = y_idx / inner_dim;
        const int iix = y_idx % inner_dim;
        const T* X = x + (oix * axis_dim * inner_dim + iix);
        T min_val = X[0], val; int64_t min_idx = 0;
        for (int j = 1; j < axis_dim; ++j) {
            val = X[j * inner_dim];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        indices[y_idx] = min_idx;
        if (values) values[y_idx] = min_val;
    }
}

/*! ArgMin <T = float16, Device = CUDA> */

__global__ void _ArgMinHalf(
    const int               nthreads,
    const int               inner_dim,
    const int               axis_dim,
    const half*             x,
    int64_t*                indices,
    half*                   values) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        const int oix = y_idx / inner_dim;
        const int iix = y_idx % inner_dim;
        const half* X = x + (oix * axis_dim * inner_dim + iix);
        half max_val = X[0], val; int64_t max_idx = 0;
        for (int j = 1; j < axis_dim; ++j) {
            val = X[j * inner_dim];
            if (__hlt(val, max_val)) {
                max_val = val;
                max_idx = j;
            }
        }
        indices[y_idx] = max_idx;
        if (values) values[y_idx] = max_val;
#endif
    }
}

/*! Kernel Launchers */

#define DEFINE_ARGREDUCE_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CUDAContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               top_k, \
        const T*                x, \
        int64_t*                indices, \
        T*                      values, \
        CUDAContext*            ctx) { \
        CHECK_EQ(top_k, 1) << "\nRequired top_k == 1."; \
        auto nthreads = outer_dim * inner_dim; \
        _##name<T> \
            << < CUDA_BLOCKS(nthreads), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (nthreads, inner_dim, axis_dim, \
                x, indices, values); \
    }

DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, bool);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, int8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, uint8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, int);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, int64_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, float);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, double);

DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, bool);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, int8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, uint8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, int);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, int64_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, float);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, double);

template<> void ArgMax<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const float16*          x,
    int64_t*                indices,
    float16*                values,
    CUDAContext*            ctx) {
    CHECK_EQ(top_k, 1) << "\nRequired top_k == 1.";
    auto nthreads = outer_dim * inner_dim;
    _ArgMaxHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, inner_dim, axis_dim,
            reinterpret_cast<const half*>(x), indices,
                reinterpret_cast<half*>(values));
}

template<> void ArgMin<float16, CUDAContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const float16*          x,
    int64_t*                indices,
    float16*                values,
    CUDAContext*            ctx) {
    CHECK_EQ(top_k, 1) << "\nRequired top_k == 1.";
    auto nthreads = outer_dim * inner_dim;
    _ArgMinHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, inner_dim, axis_dim,
            reinterpret_cast<const half*>(x), indices,
                reinterpret_cast<half*>(values));
}

#undef DEFINE_ARGREDUCE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA