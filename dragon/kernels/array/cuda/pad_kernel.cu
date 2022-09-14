#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _ConstPad(
    const int N,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_pads,
    const T value,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi, d;
#pragma unroll
    for (d = D - 1; d >= 0; --d) {
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
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_pads,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
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
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const SimpleArray<int, D> X_pads,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      r = min(X_dims.data[d] - 1, max(r - X_pads.data[d], 0));
      xi += r * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D>
void _PadImpl(
    const int64_t* x_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* pads,
    const float value,
    const string& mode,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  SimpleArray<int, D> X_dims, X_strides, Y_dims, X_pads;
  const auto N = math::utils::Prod(D, y_dims);
  for (int i = 0; i < D; ++i) {
    X_dims.data[i] = x_dims[i];
    X_strides.data[i] = x_strides[i];
    Y_dims.data[i] = y_dims[i];
    X_pads.data[i] = pads[i];
  }
  if (mode == "ConstPad") {
    _ConstPad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, X_dims, X_strides, Y_dims, X_pads, convert::To<T>(value), x, y);
  } else if (mode == "ReflectPad") {
    _ReflectPad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, X_dims, X_strides, Y_dims, X_pads, x, y);
  } else if (mode == "EdgePad") {
    _EdgePad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, X_dims, X_strides, Y_dims, X_pads, x, y);
  } else {
    LOG(FATAL) << "Unknown Pad: " << mode << ".";
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CUDAContext>(            \
      const int num_dims,               \
      const int64_t* x_dims,            \
      const int64_t* x_strides,         \
      const int64_t* y_dims,            \
      const int64_t* pads,              \
      const float value,                \
      const T* x,                       \
      T* y,                             \
      CUDAContext* ctx) {               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);   \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1( \
        _PadImpl,                       \
        T,                              \
        num_dims,                       \
        x_dims,                         \
        x_strides,                      \
        y_dims,                         \
        pads,                           \
        value,                          \
        #name,                          \
        x,                              \
        y,                              \
        ctx);                           \
  }

DEFINE_KERNEL_LAUNCHER(ConstPad, bool);
DEFINE_KERNEL_LAUNCHER(ConstPad, uint8_t);
DEFINE_KERNEL_LAUNCHER(ConstPad, int8_t);
DEFINE_KERNEL_LAUNCHER(ConstPad, int);
DEFINE_KERNEL_LAUNCHER(ConstPad, int64_t);
DEFINE_KERNEL_LAUNCHER(ConstPad, float16);
DEFINE_KERNEL_LAUNCHER(ConstPad, float);
DEFINE_KERNEL_LAUNCHER(ConstPad, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CUDAContext>(            \
      const int num_dims,               \
      const int64_t* x_dims,            \
      const int64_t* x_strides,         \
      const int64_t* y_dims,            \
      const int64_t* pads,              \
      const T* x,                       \
      T* y,                             \
      CUDAContext* ctx) {               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);   \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1( \
        _PadImpl,                       \
        T,                              \
        num_dims,                       \
        x_dims,                         \
        x_strides,                      \
        y_dims,                         \
        pads,                           \
        0.f,                            \
        #name,                          \
        x,                              \
        y,                              \
        ctx);                           \
  }

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
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
