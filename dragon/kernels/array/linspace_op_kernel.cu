#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, int D>
__global__ void _RowwiseLinSpace(
    const int nthreads,
    const int rows,
    const int cols,
    const SimpleArray<double, D> start,
    const SimpleArray<double, D> stop,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi % cols;
    const int j = yi / cols;
    if (j == rows - 1 && j > 0) {
      y[yi] = stop.data[i];
    } else {
      y[yi] = start.data[i] +
          j * ((stop.data[i] - start.data[i]) / double(rows - 1));
    }
  }
}

template <int D>
__global__ void _RowwiseLinSpace(
    const int nthreads,
    const int rows,
    const int cols,
    const SimpleArray<double, D> start,
    const SimpleArray<double, D> stop,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi % cols;
    const int j = yi / cols;
    if (j == rows - 1 && j > 0) {
      y[yi] = __float2half(float(stop.data[i]));
    } else {
      y[yi] = __float2half(float(
          start.data[i] +
          j * ((stop.data[i] - start.data[i]) / double(rows - 1))));
    }
  }
}

template <typename T, int D>
__global__ void _ColwiseLinSpace(
    const int nthreads,
    const int cols,
    const SimpleArray<double, D> start,
    const SimpleArray<double, D> stop,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / cols;
    const int j = yi % cols;
    if (j == cols - 1 && j > 0) {
      y[yi] = stop.data[i];
    } else {
      y[yi] = start.data[i] +
          j * ((stop.data[i] - start.data[i]) / double(cols - 1));
    }
  }
}

template <int D>
__global__ void _ColwiseLinSpace(
    const int nthreads,
    const int cols,
    const SimpleArray<double, D> start,
    const SimpleArray<double, D> stop,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / cols;
    const int j = yi % cols;
    if (j == cols - 1 && j > 0) {
      y[yi] = __float2half(float(stop.data[i]));
    } else {
      y[yi] = __float2half(float(
          start.data[i] +
          j * ((stop.data[i] - start.data[i]) / double(cols - 1))));
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void LinSpace<float16, CUDAContext>(
    const int rows,
    const int cols,
    const int axis,
    const double* start,
    const double* stop,
    float16* y,
    CUDAContext* ctx) {
  CUDA_TENSOR_DIMS_CHECK((axis == 0 ? cols : rows));
  const auto nthreads = rows * cols;
  SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_start;
  SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_stop;
  for (int i = 0; i < (axis == 0 ? cols : rows); ++i) {
    Y_start.data[i] = start[i];
    Y_stop.data[i] = stop[i];
  }
  if (axis == 0) {
    _RowwiseLinSpace<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads, rows, cols, Y_start, Y_stop, reinterpret_cast<half*>(y));
  } else {
    _ColwiseLinSpace<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads, cols, Y_start, Y_stop, reinterpret_cast<half*>(y));
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void LinSpace<T, CUDAContext>(                                           \
      const int rows,                                                      \
      const int cols,                                                      \
      const int axis,                                                      \
      const double* start,                                                 \
      const double* stop,                                                  \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    CUDA_TENSOR_DIMS_CHECK((axis == 0 ? cols : rows));                     \
    const auto nthreads = rows * cols;                                     \
    SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_start;                     \
    SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_stop;                      \
    for (int i = 0; i < (axis == 0 ? cols : rows); ++i) {                  \
      Y_start.data[i] = start[i];                                          \
      Y_stop.data[i] = stop[i];                                            \
    }                                                                      \
    if (axis == 0) {                                                       \
      _RowwiseLinSpace<<<                                                  \
          CUDA_BLOCKS(nthreads),                                           \
          CUDA_THREADS,                                                    \
          0,                                                               \
          ctx->cuda_stream()>>>(nthreads, rows, cols, Y_start, Y_stop, y); \
    } else {                                                               \
      _ColwiseLinSpace<<<                                                  \
          CUDA_BLOCKS(nthreads),                                           \
          CUDA_THREADS,                                                    \
          0,                                                               \
          ctx->cuda_stream()>>>(nthreads, cols, Y_start, Y_stop, y);       \
    }                                                                      \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
