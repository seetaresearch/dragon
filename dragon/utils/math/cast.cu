#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename InputT, typename OutputT>
__global__ void _Cast(const int nthreads, const InputT* x, OutputT* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = convert::To<OutputT>(x[i]);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_CAST_KERNEL_LAUNCHER(InputT, OutputT)                \
  template <>                                                       \
  DRAGON_API void Cast<InputT, OutputT, CUDAContext>(               \
      const int n, const InputT* x, OutputT* y, CUDAContext* ctx) { \
    _Cast<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n,                                                          \
        reinterpret_cast<const ScalarType<InputT>::type*>(x),       \
        reinterpret_cast<ScalarType<OutputT>::type*>(y));           \
  }

#define DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(InputT, OutputT)             \
  template <>                                                           \
  DRAGON_API void Cast<InputT, OutputT, CUDAContext>(                   \
      const int n, const InputT* x, OutputT* y, CUDAContext* ctx) {     \
    LOG(FATAL) << "Unsupported conversion: "                            \
               << types::to_string(TypeMeta::Make<InputT>()) << " -> "  \
               << types::to_string(TypeMeta::Make<OutputT>());          \
  }                                                                     \
  template <>                                                           \
  DRAGON_API void Cast<OutputT, InputT, CUDAContext>(                   \
      const int n, const OutputT* x, InputT* y, CUDAContext* ctx) {     \
    LOG(FATAL) << "Unsupported conversion: "                            \
               << types::to_string(TypeMeta::Make<OutputT>()) << " -> " \
               << types::to_string(TypeMeta::Make<InputT>());           \
  }

#define DEFINE_KERNEL_LAUNCHER(T)          \
  DEFINE_CAST_KERNEL_LAUNCHER(T, bool);    \
  DEFINE_CAST_KERNEL_LAUNCHER(T, int8_t);  \
  DEFINE_CAST_KERNEL_LAUNCHER(T, uint8_t); \
  DEFINE_CAST_KERNEL_LAUNCHER(T, int);     \
  DEFINE_CAST_KERNEL_LAUNCHER(T, int64_t); \
  DEFINE_CAST_KERNEL_LAUNCHER(T, float);   \
  DEFINE_CAST_KERNEL_LAUNCHER(T, double);

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_CAST_KERNEL_LAUNCHER(float16, float16);
DEFINE_CAST_KERNEL_LAUNCHER(float16, float);
DEFINE_CAST_KERNEL_LAUNCHER(float, float16);
DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(bool, float16);
DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(int8_t, float16);
DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(uint8_t, float16);
DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(int, float16);
DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(double, float16);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_CAST_KERNEL_LAUNCHER
#undef DEFINE_UNSUPPORTED_KERNEL_LAUNCHER

} // namespace math

} // namespace dragon

#endif // USE_CUDA
