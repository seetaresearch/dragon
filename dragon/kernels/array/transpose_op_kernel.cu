#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Transpose(
    const int N,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += r * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D>
void _TransposeImpl(
    const int N,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  SimpleArray<int, D> X_strides, Y_dims;
  for (int i = 0; i < D; ++i) {
    X_strides.data[i] = x_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _Transpose<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, X_strides, Y_dims, x, y);
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                  \
  template <>                                                      \
  void Transpose<T, CUDAContext>(                                  \
      const int num_dims,                                          \
      const int64_t* x_strides,                                    \
      const int64_t* y_dims,                                       \
      const T* x,                                                  \
      T* y,                                                        \
      CUDAContext* ctx) {                                          \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                              \
    const auto N = std::accumulate(                                \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>()); \
    switch (num_dims) {                                            \
      case 1:                                                      \
        _TransposeImpl<T, 1>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 2:                                                      \
        _TransposeImpl<T, 2>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 3:                                                      \
        _TransposeImpl<T, 3>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 4:                                                      \
        _TransposeImpl<T, 4>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 5:                                                      \
        _TransposeImpl<T, 5>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 6:                                                      \
        _TransposeImpl<T, 6>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 7:                                                      \
        _TransposeImpl<T, 7>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      case 8:                                                      \
        _TransposeImpl<T, 8>(N, x_strides, y_dims, x, y, ctx);     \
        break;                                                     \
      default:                                                     \
        break;                                                     \
    }                                                              \
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
