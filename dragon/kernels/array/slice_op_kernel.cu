#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Slice(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_starts,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += (r + X_starts.data[d]) * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D>
__global__ void _SliceGrad(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_starts,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += (r + X_starts.data[d]) * X_strides.data[d];
    }
    dx[xi] = dy[yi];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                                 \
  template <>                                                           \
  void name<T, CUDAContext>(                                            \
      const int num_dims,                                               \
      const int64_t* x_strides,                                         \
      const int64_t* y_dims,                                            \
      const int64_t* starts,                                            \
      const T* x,                                                       \
      T* y,                                                             \
      CUDAContext* ctx) {                                               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                   \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims, X_starts; \
    const auto N = std::accumulate(                                     \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());      \
    for (int i = 0; i < num_dims; ++i) {                                \
      X_strides.data[i] = x_strides[i];                                 \
      Y_dims.data[i] = y_dims[i];                                       \
      X_starts.data[i] = starts[i];                                     \
    }                                                                   \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        N, num_dims, X_strides, Y_dims, X_starts, x, y);                \
  }

DEFINE_KERNEL_LAUNCHER(Slice, bool);
DEFINE_KERNEL_LAUNCHER(Slice, int8_t);
DEFINE_KERNEL_LAUNCHER(Slice, uint8_t);
DEFINE_KERNEL_LAUNCHER(Slice, int);
DEFINE_KERNEL_LAUNCHER(Slice, int64_t);
DEFINE_KERNEL_LAUNCHER(Slice, float16);
DEFINE_KERNEL_LAUNCHER(Slice, float);
DEFINE_KERNEL_LAUNCHER(Slice, double);
DEFINE_KERNEL_LAUNCHER(SliceGrad, float16);
DEFINE_KERNEL_LAUNCHER(SliceGrad, float);
DEFINE_KERNEL_LAUNCHER(SliceGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
