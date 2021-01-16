#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, int D>
__global__ void _Assign(
    const int nthreads,
    const int num_dims,
    const SimpleArray<int, D> x_dims,
    const SimpleArray<int, D> y_strides,
    const SimpleArray<int, D> starts,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int yi = 0, tmp = xi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(x_dims.data[d], tmp, &tmp, &r);
      yi += (r + starts.data[d]) * y_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Assign<T, CUDAContext>(                                               \
      const int num_dims,                                                    \
      const int64_t* x_dims,                                                 \
      const int64_t* y_strides,                                              \
      const int64_t* starts,                                                 \
      const T* x,                                                            \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                        \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, Y_strides, X_starts;      \
    const auto nthreads = std::accumulate(                                   \
        x_dims, x_dims + num_dims, 1, std::multiplies<int64_t>());           \
    for (int i = 0; i < num_dims; ++i) {                                     \
      X_dims.data[i] = x_dims[i];                                            \
      Y_strides.data[i] = y_strides[i];                                      \
      X_starts.data[i] = starts[i];                                          \
    }                                                                        \
    _Assign<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, num_dims, X_dims, Y_strides, X_starts, x, y);              \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
