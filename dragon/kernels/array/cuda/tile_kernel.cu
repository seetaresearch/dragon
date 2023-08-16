#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Tile(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += (r % X_dims.data[d]) * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void Tile<T, CUDAContext>(                                          \
      const int num_dims,                                             \
      const int64_t* x_dims,                                          \
      const int64_t* x_strides,                                       \
      const int64_t* y_dims,                                          \
      const T* x,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                 \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, X_strides, Y_dims; \
    const auto N = math::utils::Prod(num_dims, y_dims);               \
    for (int i = 0; i < num_dims; ++i) {                              \
      X_dims.data[i] = x_dims[i];                                     \
      X_strides.data[i] = x_strides[i];                               \
      Y_dims.data[i] = y_dims[i];                                     \
    }                                                                 \
    _Tile<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        N, num_dims, X_dims, X_strides, Y_dims, x, y);                \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
