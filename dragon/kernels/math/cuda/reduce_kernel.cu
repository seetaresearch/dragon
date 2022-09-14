#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT, int D>
__global__ void _ReduceSumGrad(
    const int N,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> Y_strides,
    const AccT scale,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, N) {
    int yi = 0, tmp = xi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(X_dims.data[d], tmp, &tmp, &r);
      yi += (r % Y_dims.data[d]) * Y_strides.data[d];
    }
    dx[xi] = convert::To<T>(convert::To<AccT>(__ldg(dy + yi)) * scale);
  }
}

template <typename T, typename AccT, int D>
void _ReduceSumGradImpl(
    const int64_t* x_dims,
    const int64_t* y_dims,
    const int64_t* y_strides,
    const AccT scale,
    const T* dy,
    T* dx,
    CUDAContext* ctx) {
  SimpleArray<int, D> X_dims, Y_dims, Y_strides;
  const auto N = math::utils::Prod(D, x_dims);
  for (int i = 0; i < D; ++i) {
    X_dims.data[i] = x_dims[i];
    Y_dims.data[i] = y_dims[i];
    Y_strides.data[i] = y_strides[i];
  }
  _ReduceSumGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, X_dims, Y_dims, Y_strides, scale, dy, dx);
}

} // namespace

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                          \
  template <>                                                   \
  void ReduceSumGrad<T, CUDAContext>(                           \
      const int num_dims,                                       \
      const int64_t* x_dims,                                    \
      const int64_t* y_dims,                                    \
      const int64_t* y_strides,                                 \
      const float scale,                                        \
      const T* dy,                                              \
      T* dx,                                                    \
      CUDAContext* ctx) {                                       \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                           \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_2(                         \
        _ReduceSumGradImpl,                                     \
        math::ScalarType<T>::type,                              \
        math::AccumulatorType<T>::type,                         \
        num_dims,                                               \
        x_dims,                                                 \
        y_dims,                                                 \
        y_strides,                                              \
        convert::To<math::AccumulatorType<T>::type>(scale),     \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
        reinterpret_cast<math::ScalarType<T>::type*>(dx),       \
        ctx);                                                   \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
