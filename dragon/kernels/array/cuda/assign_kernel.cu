#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Assign(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<int, D> X_starts,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(xi, N) {
    int yi = 0, tmp = xi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(X_dims.data[d], tmp, &tmp, &r);
      yi += (r + X_starts.data[d]) * Y_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                       \
  template <>                                                           \
  void Assign<T, CUDAContext>(                                          \
      const int num_dims,                                               \
      const int64_t* x_dims,                                            \
      const int64_t* y_strides,                                         \
      const int64_t* starts,                                            \
      const T* x,                                                       \
      T* y,                                                             \
      CUDAContext* ctx) {                                               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                   \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, Y_strides, X_starts; \
    const auto N = math::utils::Prod(num_dims, x_dims);                 \
    for (int i = 0; i < num_dims; ++i) {                                \
      X_dims.data[i] = x_dims[i];                                       \
      Y_strides.data[i] = y_strides[i];                                 \
      X_starts.data[i] = starts[i];                                     \
    }                                                                   \
    _Assign<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        N, num_dims, X_dims, Y_strides, X_starts, x, y);                \
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
