#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _ScatterElements(
    const int N,
    const int axis,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> Y_strides,
    const int64_t* index,
    const T value,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int yi = 0, tmp = i;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(X_dims.data[d], tmp, &tmp, &r);
      yi += (d == axis ? index[i] : r) * Y_strides.data[d];
    }
    y[yi] = value;
  }
}

template <typename T, int D>
__global__ void _ScatterElements(
    const int N,
    const int axis,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_strides,
    const int64_t* index,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int xi = 0, yi = 0, tmp = i;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(X_dims.data[d], tmp, &tmp, &r);
      xi += r * X_strides.data[d];
      yi += (d == axis ? index[i] : r) * Y_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, typename AccT, int D>
__global__ void _ScatterAdd(
    const int N,
    const int axis,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_strides,
    const int64_t* index,
    const T* x,
    AccT* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int xi = 0, yi = 0, tmp = i;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(X_dims.data[d], tmp, &tmp, &r);
      xi += r * X_strides.data[d];
      yi += (d == axis ? index[i] : r) * Y_strides.data[d];
    }
    math::utils::AtomicAdd(y + yi, convert::To<AccT>(x[xi]));
  }
}

template <typename T, int D>
void DispatchScatter(
    const int axis,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  const auto N = math::utils::Prod(D, dims);
  SimpleArray<int, D> X_dims, X_strides, Y_strides;
  for (int i = 0; i < D; ++i) {
    X_dims.data[i] = dims[i];
    X_strides.data[i] = (x_strides != nullptr ? x_strides[i] : 0);
    Y_strides.data[i] = y_strides[i];
  }
  if (x_strides == nullptr) {
    _ScatterElements<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, axis, X_dims, Y_strides, index, x[0], y);
  } else {
    _ScatterElements<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, axis, X_dims, X_strides, Y_strides, index, x, y);
  }
}

template <typename InputT, typename OutputT, int D>
void DispatchScatterAcc(
    const string& kernel,
    const int axis,
    const int64_t* dims,
    const int64_t* x_strides,
    const int64_t* y_strides,
    const int64_t* index,
    const InputT* x,
    OutputT* y,
    CUDAContext* ctx) {
  const auto N = math::utils::Prod(D, dims);
  SimpleArray<int, D> X_dims, X_strides, Y_strides;
  for (int i = 0; i < D; ++i) {
    X_dims.data[i] = dims[i];
    X_strides.data[i] = x_strides[i];
    Y_strides.data[i] = y_strides[i];
  }
  using T = typename math::Traits<InputT>::scalar_type;
  using AccT = typename math::Traits<OutputT>::scalar_type;
  if (kernel == "ScatterAdd") {
    _ScatterAdd<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        axis,
        X_dims,
        X_strides,
        Y_strides,
        index,
        reinterpret_cast<const T*>(x),
        reinterpret_cast<AccT*>(y));
  } else {
    NOT_IMPLEMENTED;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CUDAContext>(            \
      const int axis,                   \
      const int num_dims,               \
      const T value,                    \
      const int64_t* dims,              \
      const int64_t* y_strides,         \
      const int64_t* index,             \
      T* y,                             \
      CUDAContext* ctx) {               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);   \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1( \
        DispatchScatter,                \
        T,                              \
        num_dims,                       \
        axis,                           \
        dims,                           \
        nullptr,                        \
        y_strides,                      \
        index,                          \
        &value,                         \
        y,                              \
        ctx);                           \
  }

DEFINE_KERNEL_LAUNCHER(ScatterElements, bool);
DEFINE_KERNEL_LAUNCHER(ScatterElements, uint8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int64_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, bfloat16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float);
DEFINE_KERNEL_LAUNCHER(ScatterElements, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CUDAContext>(            \
      const int axis,                   \
      const int num_dims,               \
      const int64_t* dims,              \
      const int64_t* x_strides,         \
      const int64_t* y_strides,         \
      const int64_t* index,             \
      const T* x,                       \
      T* y,                             \
      CUDAContext* ctx) {               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);   \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1( \
        DispatchScatter,                \
        T,                              \
        num_dims,                       \
        axis,                           \
        dims,                           \
        x_strides,                      \
        y_strides,                      \
        index,                          \
        x,                              \
        y,                              \
        ctx);                           \
  }

DEFINE_KERNEL_LAUNCHER(ScatterElements, bool);
DEFINE_KERNEL_LAUNCHER(ScatterElements, uint8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int8_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int);
DEFINE_KERNEL_LAUNCHER(ScatterElements, int64_t);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, bfloat16);
DEFINE_KERNEL_LAUNCHER(ScatterElements, float);
DEFINE_KERNEL_LAUNCHER(ScatterElements, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, InputT, OutputT) \
  template <>                                         \
  void name<InputT, OutputT, CUDAContext>(            \
      const int axis,                                 \
      const int num_dims,                             \
      const int64_t* dims,                            \
      const int64_t* x_strides,                       \
      const int64_t* y_strides,                       \
      const int64_t* index,                           \
      const InputT* x,                                \
      OutputT* y,                                     \
      CUDAContext* ctx) {                             \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                 \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_2(               \
        DispatchScatterAcc,                           \
        InputT,                                       \
        OutputT,                                      \
        num_dims,                                     \
        #name,                                        \
        axis,                                         \
        dims,                                         \
        x_strides,                                    \
        y_strides,                                    \
        index,                                        \
        x,                                            \
        y,                                            \
        ctx);                                         \
  }

DEFINE_KERNEL_LAUNCHER(ScatterAdd, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, int, int)
DEFINE_KERNEL_LAUNCHER(ScatterAdd, int64_t, int64_t)
DEFINE_KERNEL_LAUNCHER(ScatterAdd, float16, float);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, bfloat16, float);
DEFINE_KERNEL_LAUNCHER(ScatterAdd, float, float)
DEFINE_KERNEL_LAUNCHER(ScatterAdd, double, float);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
