#include "dragon/kernels/training/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _CheckFinite(const int N, const T* g, float* isinf) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    if (!math::utils::IsFinite(g[i])) *isinf = 1.f;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                      \
  template <>                                                                \
  void name<T, CUDAContext>(                                                 \
      const int N, const T* g, float* isinf, CUDAContext* ctx) {             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
        N, reinterpret_cast<const math::Traits<T>::scalar_type*>(g), isinf); \
  }

DEFINE_KERNEL_LAUNCHER(CheckFinite, float16);
DEFINE_KERNEL_LAUNCHER(CheckFinite, bfloat16);
DEFINE_KERNEL_LAUNCHER(CheckFinite, float);
DEFINE_KERNEL_LAUNCHER(CheckFinite, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
