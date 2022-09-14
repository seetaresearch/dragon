#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Gather(
    const int NxKxS,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, NxKxS) {
    const int j = yi % S;
    const int i = yi / S / K;
    int pos = __ldg(index + yi / S % K);
    pos = (pos >= 0 ? pos : pos + C);
    y[yi] = x[(i * C + pos) * S + j];
  }
}

template <typename T, typename AccT>
__global__ void _GatherGrad(
    const int NxKxS,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* dy,
    AccT* dx) {
  CUDA_1D_KERNEL_LOOP(yi, NxKxS) {
    const int j = yi % S;
    const int i = yi / S / K;
    int pos = __ldg(index + yi / S % K);
    pos = (pos >= 0 ? pos : pos + C);
    math::utils::AtomicAdd(
        dx + (i * C + pos) * S + j, convert::To<AccT>(dy[yi]));
  }
}

template <typename T, int D>
__global__ void _GatherElements(
    const int N,
    const int axis,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const int64_t* index,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += (d == axis ? index[yi] : r) * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D>
void _GatherElementsImpl(
    const int axis,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const int64_t* index,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  const auto N = math::utils::Prod(D, y_dims);
  SimpleArray<int, D> X_strides, Y_dims;
  for (int i = 0; i < D; ++i) {
    X_strides.data[i] = x_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _GatherElements<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, axis, X_strides, Y_dims, index, x, y);
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, InputT, OutputT)                     \
  template <>                                                             \
  void name<InputT, CUDAContext>(                                         \
      const int N,                                                        \
      const int S,                                                        \
      const int C,                                                        \
      const int K,                                                        \
      const int64_t* index,                                               \
      const InputT* x,                                                    \
      OutputT* y,                                                         \
      CUDAContext* ctx) {                                                 \
    const int NxKxS = N * K * S;                                          \
    _##name<<<CUDA_BLOCKS(NxKxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxKxS,                                                            \
        S,                                                                \
        C,                                                                \
        K,                                                                \
        index,                                                            \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(x),       \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));           \
  }

DEFINE_KERNEL_LAUNCHER(Gather, bool, bool);
DEFINE_KERNEL_LAUNCHER(Gather, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(Gather, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(Gather, int, int);
DEFINE_KERNEL_LAUNCHER(Gather, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(Gather, float16, float16);
DEFINE_KERNEL_LAUNCHER(Gather, float, float);
DEFINE_KERNEL_LAUNCHER(Gather, double, double);
DEFINE_KERNEL_LAUNCHER(GatherGrad, float16, float); // GatherGrad
DEFINE_KERNEL_LAUNCHER(GatherGrad, float, float); // GatherGrad
DEFINE_KERNEL_LAUNCHER(GatherGrad, double, float); // GatherGrad
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CUDAContext>(            \
      const int axis,                   \
      const int num_dims,               \
      const int64_t* x_strides,         \
      const int64_t* y_dims,            \
      const int64_t* index,             \
      const T* x,                       \
      T* y,                             \
      CUDAContext* ctx) {               \
    CUDA_TENSOR_DIMS_CHECK(num_dims);   \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1( \
        _GatherElementsImpl,            \
        T,                              \
        num_dims,                       \
        axis,                           \
        x_strides,                      \
        y_dims,                         \
        index,                          \
        x,                              \
        y,                              \
        ctx);                           \
  }

DEFINE_KERNEL_LAUNCHER(GatherElements, bool);
DEFINE_KERNEL_LAUNCHER(GatherElements, uint8_t);
DEFINE_KERNEL_LAUNCHER(GatherElements, int8_t);
DEFINE_KERNEL_LAUNCHER(GatherElements, int);
DEFINE_KERNEL_LAUNCHER(GatherElements, int64_t);
DEFINE_KERNEL_LAUNCHER(GatherElements, float16);
DEFINE_KERNEL_LAUNCHER(GatherElements, float);
DEFINE_KERNEL_LAUNCHER(GatherElements, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
