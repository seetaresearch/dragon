#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _ArgMax(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    int64_t* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / inner_dim;
    const int j = yi % inner_dim;
    const T* offset_x = x + (i * axis_dim * inner_dim + j);
    auto max_val = offset_x[0];
    auto max_idx = int64_t(0);
    for (int k = 1; k < axis_dim; ++k) {
      const T val = offset_x[k * inner_dim];
      if (val > max_val) {
        max_val = val;
        max_idx = k;
      }
    }
    y[yi] = max_idx;
  }
}

template <>
__global__ void _ArgMax<half>(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const half* x,
    int64_t* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / inner_dim;
    const int j = yi % inner_dim;
    const half* offset_x = x + (i * axis_dim * inner_dim + j);
    auto max_val = __half2float(offset_x[0]);
    auto max_idx = int64_t(0);
    for (int k = 1; k < axis_dim; ++k) {
      const float val = __half2float(offset_x[k * inner_dim]);
      if (val > max_val) {
        max_val = val;
        max_idx = k;
      }
    }
    y[yi] = max_idx;
  }
}

template <typename T>
__global__ void _ArgMin(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    int64_t* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / inner_dim;
    const int j = yi % inner_dim;
    const T* offset_x = x + (i * axis_dim * inner_dim + j);
    auto min_val = offset_x[0];
    auto min_idx = int64_t(0);
    for (int k = 1; k < axis_dim; ++k) {
      const T val = offset_x[k * inner_dim];
      if (val < min_val) {
        min_val = val;
        min_idx = k;
      }
    }
    y[yi] = min_idx;
  }
}

template <>
__global__ void _ArgMin<half>(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const half* x,
    int64_t* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / inner_dim;
    const int j = yi % inner_dim;
    const half* offset_x = x + (i * axis_dim * inner_dim + j);
    auto min_val = __half2float(offset_x[0]);
    auto min_idx = int64_t(0);
    for (int k = 1; k < axis_dim; ++k) {
      const float val = __half2float(offset_x[k * inner_dim]);
      if (val < min_val) {
        min_val = val;
        min_idx = k;
      }
    }
    y[yi] = min_idx;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T1, T2)                                 \
  template <>                                                                \
  void name<T1, CUDAContext>(                                                \
      const int outer_dim,                                                   \
      const int inner_dim,                                                   \
      const int axis_dim,                                                    \
      const T1* x,                                                           \
      int64_t* y,                                                            \
      CUDAContext* ctx) {                                                    \
    auto nthreads = outer_dim * inner_dim;                                   \
    _##name<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, inner_dim, axis_dim, reinterpret_cast<const T2*>(x), y);   \
  }

DEFINE_KERNEL_LAUNCHER(ArgMax, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, int, int);
DEFINE_KERNEL_LAUNCHER(ArgMax, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMax, float16, half);
DEFINE_KERNEL_LAUNCHER(ArgMax, float, float);
DEFINE_KERNEL_LAUNCHER(ArgMax, double, double);
DEFINE_KERNEL_LAUNCHER(ArgMin, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, int, int);
DEFINE_KERNEL_LAUNCHER(ArgMin, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ArgMin, float16, half);
DEFINE_KERNEL_LAUNCHER(ArgMin, float, float);
DEFINE_KERNEL_LAUNCHER(ArgMin, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
