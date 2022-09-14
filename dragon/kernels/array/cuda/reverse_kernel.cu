#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Reverse(
    const int N,
    const int num_dims,
    const SimpleArray<uint8_t, D> X_flips,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += (X_flips.data[d] ? Y_dims.data[d] - r - 1 : r) * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void Reverse<T, CUDAContext>(                                        \
      const int num_dims,                                              \
      const uint8_t* x_flips,                                          \
      const int64_t* x_strides,                                        \
      const int64_t* y_dims,                                           \
      const T* x,                                                      \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                  \
    SimpleArray<uint8_t, CUDA_TENSOR_MAX_DIMS> X_flips;                \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;          \
    const auto N = math::utils::Prod(num_dims, y_dims);                \
    for (int i = 0; i < num_dims; ++i) {                               \
      X_flips.data[i] = x_flips[i];                                    \
      X_strides.data[i] = x_strides[i];                                \
      Y_dims.data[i] = y_dims[i];                                      \
    }                                                                  \
    _Reverse<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, num_dims, X_flips, X_strides, Y_dims, x, y);                \
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
