#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

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

template <typename T>
__global__ void _GatherGrad(
    const int NxKxS,
    const int S,
    const int C,
    const int K,
    const int64_t* index,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, NxKxS) {
    const int j = yi % S;
    const int i = yi / S / K;
    int pos = __ldg(index + yi / S % K);
    pos = (pos >= 0 ? pos : pos + C);
    math::utils::AtomicAdd(
        dx + (i * C + pos) * S + j, convert::To<float>(dy[yi]));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

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

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
