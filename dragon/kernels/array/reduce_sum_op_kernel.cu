#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _ReduceSumGrad(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> Y_strides,
    const float scale,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, N) {
    int yi = 0, tmp = xi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(X_dims.data[d], tmp, &tmp, &r);
      yi += (r % Y_dims.data[d]) * Y_strides.data[d];
    }
    dx[xi] = convert::To<T>(convert::To<float>(__ldg(dy + yi)) * scale);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void ReduceSumGrad<T, CUDAContext>(                                        \
      const int num_dims,                                                    \
      const int64_t* x_dims,                                                 \
      const int64_t* y_dims,                                                 \
      const int64_t* y_strides,                                              \
      const float scale,                                                     \
      const T* dy,                                                           \
      T* dx,                                                                 \
      CUDAContext* ctx) {                                                    \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                        \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, Y_dims, Y_strides;        \
    const auto N = std::accumulate(                                          \
        x_dims, x_dims + num_dims, 1, std::multiplies<int64_t>());           \
    for (int i = 0; i < num_dims; ++i) {                                     \
      X_dims.data[i] = x_dims[i];                                            \
      Y_dims.data[i] = y_dims[i];                                            \
      Y_strides.data[i] = y_strides[i];                                      \
    }                                                                        \
    _ReduceSumGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                   \
        num_dims,                                                            \
        X_dims,                                                              \
        Y_dims,                                                              \
        Y_strides,                                                           \
        scale,                                                               \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),              \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));                   \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
