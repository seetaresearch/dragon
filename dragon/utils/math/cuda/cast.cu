#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename InputT, typename OutputT>
__global__ void _Cast(const int N, const InputT* x, OutputT* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<OutputT>(x[i]);
  }
}

} // namespace

#define DEFINE_CAST_FUNC(InputT, OutputT)                           \
  template <>                                                       \
  DRAGON_API void Cast<InputT, OutputT, CUDAContext>(               \
      const int N, const InputT* x, OutputT* y, CUDAContext* ctx) { \
    _Cast<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                          \
        reinterpret_cast<const ScalarType<InputT>::type*>(x),       \
        reinterpret_cast<ScalarType<OutputT>::type*>(y));           \
  }

#define DEFINE_UNSUPPORTED_CAST_FUNC(InputT, OutputT)                    \
  template <>                                                            \
  DRAGON_API void Cast<InputT, OutputT, CUDAContext>(                    \
      const int N, const InputT* x, OutputT* y, CUDAContext* ctx) {      \
    LOG(FATAL) << "Unsupported conversion: "                             \
               << dtypes::to_string(TypeMeta::Make<InputT>()) << " -> "  \
               << dtypes::to_string(TypeMeta::Make<OutputT>());          \
  }                                                                      \
  template <>                                                            \
  DRAGON_API void Cast<OutputT, InputT, CUDAContext>(                    \
      const int N, const OutputT* x, InputT* y, CUDAContext* ctx) {      \
    LOG(FATAL) << "Unsupported conversion: "                             \
               << dtypes::to_string(TypeMeta::Make<OutputT>()) << " -> " \
               << dtypes::to_string(TypeMeta::Make<InputT>());           \
  }

#define DEFINE_CAST_FUNC_TO(T)  \
  DEFINE_CAST_FUNC(T, bool);    \
  DEFINE_CAST_FUNC(T, uint8_t); \
  DEFINE_CAST_FUNC(T, int8_t);  \
  DEFINE_CAST_FUNC(T, int);     \
  DEFINE_CAST_FUNC(T, int64_t); \
  DEFINE_CAST_FUNC(T, float);   \
  DEFINE_CAST_FUNC(T, double);

DEFINE_CAST_FUNC_TO(bool);
DEFINE_CAST_FUNC_TO(uint8_t);
DEFINE_CAST_FUNC_TO(int8_t);
DEFINE_CAST_FUNC_TO(int);
DEFINE_CAST_FUNC_TO(int64_t);
DEFINE_CAST_FUNC_TO(float);
DEFINE_CAST_FUNC_TO(double);
DEFINE_CAST_FUNC(float16, float16);
DEFINE_CAST_FUNC(float16, float);
DEFINE_CAST_FUNC(float, float16);
DEFINE_UNSUPPORTED_CAST_FUNC(bool, float16);
DEFINE_UNSUPPORTED_CAST_FUNC(uint8_t, float16);
DEFINE_UNSUPPORTED_CAST_FUNC(int8_t, float16);
DEFINE_UNSUPPORTED_CAST_FUNC(int, float16);
DEFINE_UNSUPPORTED_CAST_FUNC(int64_t, float16);
DEFINE_UNSUPPORTED_CAST_FUNC(double, float16);
#undef DEFINE_CAST_FUNC
#undef DEFINE_UNSUPPORTED_CAST_FUNC
#undef DEFINE_CAST_FUNC_TO

} // namespace math

} // namespace dragon
