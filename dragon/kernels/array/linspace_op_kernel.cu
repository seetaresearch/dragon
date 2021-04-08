#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _RowwiseLinSpace(
    const int N,
    const int C,
    const SimpleArray<double, D> Y_starts,
    const SimpleArray<double, D> Y_stops,
    T* y) {
  const auto NxC = N * C;
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int i = index % C;
    const int j = index / C;
    if (j == N - 1 && j > 0) {
      y[index] = convert::To<T>(Y_stops.data[i]);
    } else {
      y[index] = convert::To<T>(
          Y_starts.data[i] +
          j * ((Y_stops.data[i] - Y_starts.data[i]) / double(N - 1)));
    }
  }
}

template <typename T, int D>
__global__ void _ColwiseLinSpace(
    const int NxC,
    const int C,
    const SimpleArray<double, D> Y_starts,
    const SimpleArray<double, D> Y_stops,
    T* y) {
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int i = index / C;
    const int j = index % C;
    if (j == C - 1 && j > 0) {
      y[index] = convert::To<T>(Y_stops.data[i]);
    } else {
      y[index] = convert::To<T>(
          Y_starts.data[i] +
          j * ((Y_stops.data[i] - Y_starts.data[i]) / double(C - 1)));
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                           \
  template <>                                               \
  void LinSpace<T, CUDAContext>(                            \
      const int N,                                          \
      const int C,                                          \
      const int axis,                                       \
      const double* starts,                                 \
      const double* stops,                                  \
      T* y,                                                 \
      CUDAContext* ctx) {                                   \
    CUDA_TENSOR_DIMS_CHECK((axis == 0 ? C : N));            \
    const auto NxC = N * C;                                 \
    SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_starts;     \
    SimpleArray<double, CUDA_TENSOR_MAX_DIMS> Y_stops;      \
    for (int i = 0; i < (axis == 0 ? C : N); ++i) {         \
      Y_starts.data[i] = starts[i];                         \
      Y_stops.data[i] = stops[i];                           \
    }                                                       \
    if (axis == 0) {                                        \
      _RowwiseLinSpace<<<                                   \
          CUDA_BLOCKS(NxC),                                 \
          CUDA_THREADS,                                     \
          0,                                                \
          ctx->cuda_stream()>>>(                            \
          N,                                                \
          C,                                                \
          Y_starts,                                         \
          Y_stops,                                          \
          reinterpret_cast<math::ScalarType<T>::type*>(y)); \
    } else {                                                \
      _ColwiseLinSpace<<<                                   \
          CUDA_BLOCKS(NxC),                                 \
          CUDA_THREADS,                                     \
          0,                                                \
          ctx->cuda_stream()>>>(                            \
          NxC,                                              \
          C,                                                \
          Y_starts,                                         \
          Y_stops,                                          \
          reinterpret_cast<math::ScalarType<T>::type*>(y)); \
    }                                                       \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
