#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _CumSum(
    const int NxS,
    const int S,
    const int C,
    const bool exclusive,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxS) {
    int offset = i / S * C * S + i % S;
    y[offset] = exclusive ? convert::To<T>(AccT(0)) : x[offset];
    for (int j = 1; j < C; ++j) {
      const int index = offset + S;
      y[index] = convert::To<AccT>(y[offset]) +
          convert::To<AccT>(x[exclusive ? offset : index]);
      offset = index;
    }
  }
}

template <typename T, typename AccT>
__global__ void _CumSumReverse(
    const int NxS,
    const int S,
    const int C,
    const bool exclusive,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxS) {
    int offset = (i / S * C + C - 1) * S + i % S;
    y[offset] = exclusive ? convert::To<T>(AccT(0)) : x[offset];
    for (int j = C - 2; j >= 0; --j) {
      const int index = offset - S;
      y[index] = convert::To<AccT>(y[offset]) +
          convert::To<AccT>(x[exclusive ? offset : index]);
      offset = index;
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void CumSum<T, CUDAContext>(                                         \
      const int N,                                                     \
      const int S,                                                     \
      const int C,                                                     \
      const bool exclusive,                                            \
      const bool reverse,                                              \
      const T* x,                                                      \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    using ScalarT = math::Traits<T>::scalar_type;                      \
    using AccT = math::Traits<T>::accumulator_type;                    \
    const auto NxS = N * S;                                            \
    if (reverse) {                                                     \
      _CumSumReverse<ScalarT, AccT>                                    \
          <<<CUDA_BLOCKS(NxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              NxS, S, C, exclusive, (const ScalarT*)x, (ScalarT*)y);   \
    } else {                                                           \
      _CumSum<ScalarT, AccT>                                           \
          <<<CUDA_BLOCKS(NxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              NxS, S, C, exclusive, (const ScalarT*)x, (ScalarT*)y);   \
    }                                                                  \
  }

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
