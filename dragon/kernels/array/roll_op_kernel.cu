#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Roll(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_shifts,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      r -= X_shifts.data[d];
      r = (r < 0 ? r + Y_dims.data[d] : r) % Y_dims.data[d];
      xi += r * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                   \
  template <>                                                       \
  void Roll<T, CUDAContext>(                                        \
      const int num_dims,                                           \
      const int64_t* x_shifts,                                      \
      const int64_t* x_strides,                                     \
      const int64_t* y_dims,                                        \
      const T* x,                                                   \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                               \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_shifts;                \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides;               \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> Y_dims;                  \
    const auto N = std::accumulate(                                 \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());  \
    for (int i = 0; i < num_dims; ++i) {                            \
      X_shifts.data[i] = x_shifts[i];                               \
      X_strides.data[i] = x_strides[i];                             \
      Y_dims.data[i] = y_dims[i];                                   \
    }                                                               \
    _Roll<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, num_dims, X_shifts, X_strides, Y_dims, x, y);            \
  }

DEFINE_KERNEL_LAUNCHER(bool);
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
