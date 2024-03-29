#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _ConstPad(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_pads,
    const T value,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi, d;
    for (d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      r -= X_pads.data[d];
      if (r < 0 || r >= X_dims.data[d]) break;
      xi += r * X_strides.data[d];
    }
    y[yi] = d >= 0 ? value : x[xi];
  }
}

template <typename T, int D>
__global__ void _ReflectPad(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_pads,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      r -= X_pads.data[d];
      r = max(r, -r);
      r = min(r, 2 * X_dims.data[d] - r - 2);
      xi += r * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D>
__global__ void _EdgePad(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_pads,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      r = min(X_dims.data[d] - 1, max(r - X_pads.data[d], 0));
      xi += r * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_CONST_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                 \
  void ConstPad<T, CUDAContext>(                                              \
      const int num_dims,                                                     \
      const int64_t* x_dims,                                                  \
      const int64_t* x_strides,                                               \
      const int64_t* y_dims,                                                  \
      const int64_t* pads,                                                    \
      const float value,                                                      \
      const T* x,                                                             \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                         \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, X_strides, Y_dims, X_pads; \
    const auto N = std::accumulate(                                           \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());            \
    for (int i = 0; i < num_dims; ++i) {                                      \
      X_dims.data[i] = x_dims[i];                                             \
      X_strides.data[i] = x_strides[i];                                       \
      Y_dims.data[i] = y_dims[i];                                             \
      X_pads.data[i] = pads[i];                                               \
    }                                                                         \
    _ConstPad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(       \
        N,                                                                    \
        num_dims,                                                             \
        X_dims,                                                               \
        X_strides,                                                            \
        Y_dims,                                                               \
        X_pads,                                                               \
        convert::To<T>(value),                                                \
        x,                                                                    \
        y);                                                                   \
  }

#define DEFINE_KERNEL_LAUNCHER(name, T)                                       \
  template <>                                                                 \
  void name<T, CUDAContext>(                                                  \
      const int num_dims,                                                     \
      const int64_t* x_dims,                                                  \
      const int64_t* x_strides,                                               \
      const int64_t* y_dims,                                                  \
      const int64_t* pads,                                                    \
      const T* x,                                                             \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                         \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, X_strides, Y_dims, X_pads; \
    const auto N = std::accumulate(                                           \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());            \
    for (int i = 0; i < num_dims; ++i) {                                      \
      X_dims.data[i] = x_dims[i];                                             \
      X_strides.data[i] = x_strides[i];                                       \
      Y_dims.data[i] = y_dims[i];                                             \
      X_pads.data[i] = pads[i];                                               \
    }                                                                         \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(         \
        N, num_dims, X_dims, X_strides, Y_dims, X_pads, x, y);                \
  }

DEFINE_CONST_KERNEL_LAUNCHER(bool);
DEFINE_CONST_KERNEL_LAUNCHER(uint8_t);
DEFINE_CONST_KERNEL_LAUNCHER(int8_t);
DEFINE_CONST_KERNEL_LAUNCHER(int);
DEFINE_CONST_KERNEL_LAUNCHER(int64_t);
DEFINE_CONST_KERNEL_LAUNCHER(float16);
DEFINE_CONST_KERNEL_LAUNCHER(float);
DEFINE_CONST_KERNEL_LAUNCHER(double);
DEFINE_KERNEL_LAUNCHER(ReflectPad, bool);
DEFINE_KERNEL_LAUNCHER(ReflectPad, uint8_t);
DEFINE_KERNEL_LAUNCHER(ReflectPad, int8_t);
DEFINE_KERNEL_LAUNCHER(ReflectPad, int);
DEFINE_KERNEL_LAUNCHER(ReflectPad, int64_t);
DEFINE_KERNEL_LAUNCHER(ReflectPad, float16);
DEFINE_KERNEL_LAUNCHER(ReflectPad, float);
DEFINE_KERNEL_LAUNCHER(ReflectPad, double);
DEFINE_KERNEL_LAUNCHER(EdgePad, bool);
DEFINE_KERNEL_LAUNCHER(EdgePad, uint8_t);
DEFINE_KERNEL_LAUNCHER(EdgePad, int8_t);
DEFINE_KERNEL_LAUNCHER(EdgePad, int);
DEFINE_KERNEL_LAUNCHER(EdgePad, int64_t);
DEFINE_KERNEL_LAUNCHER(EdgePad, float16);
DEFINE_KERNEL_LAUNCHER(EdgePad, float);
DEFINE_KERNEL_LAUNCHER(EdgePad, double);
#undef DEFINE_CONST_KERNEL_LAUNCHER
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
