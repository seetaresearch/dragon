#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, int D>
__global__ void _Transpose(
    const int nthreads,
    const int ndims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, tmp = yi;
    for (int d = ndims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D>
__global__ void _TransposeGrad(
    const int nthreads,
    const int ndims,
    const SimpleArray<int, D> x_strides,
    const SimpleArray<int, D> y_dims,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int xi = 0, tmp = yi;
    for (int d = ndims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(y_dims.data[d], tmp, &tmp, &r);
      xi += r * x_strides.data[d];
    }
    dx[xi] = dy[yi];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                                      \
  template <>                                                                \
  void name<T, CUDAContext>(                                                 \
      const int num_dims,                                                    \
      const int64_t* x_strides,                                              \
      const int64_t* y_dims,                                                 \
      const T* x,                                                            \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                        \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_strides, Y_dims;                \
    const auto nthreads = std::accumulate(                                   \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());           \
    for (int i = 0; i < num_dims; ++i) {                                     \
      X_strides.data[i] = x_strides[i];                                      \
      Y_dims.data[i] = y_dims[i];                                            \
    }                                                                        \
    _##name<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, num_dims, X_strides, Y_dims, x, y);                        \
  }

DEFINE_KERNEL_LAUNCHER(Transpose, bool);
DEFINE_KERNEL_LAUNCHER(Transpose, int8_t);
DEFINE_KERNEL_LAUNCHER(Transpose, uint8_t);
DEFINE_KERNEL_LAUNCHER(Transpose, int);
DEFINE_KERNEL_LAUNCHER(Transpose, int64_t);
DEFINE_KERNEL_LAUNCHER(Transpose, float16);
DEFINE_KERNEL_LAUNCHER(Transpose, float);
DEFINE_KERNEL_LAUNCHER(Transpose, double);

DEFINE_KERNEL_LAUNCHER(TransposeGrad, float16);
DEFINE_KERNEL_LAUNCHER(TransposeGrad, float);
DEFINE_KERNEL_LAUNCHER(TransposeGrad, double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
