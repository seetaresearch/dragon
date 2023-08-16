#include "dragon/core/workspace.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/random.h"

namespace dragon {

namespace math {

template <>
DRAGON_API void
Random<uint32_t, CUDAContext>(const int N, uint32_t* y, CUDAContext* ctx) {
  CURAND_CHECK(curandGenerate(ctx->curand_generator(), y, N));
}

template <>
DRAGON_API void RandomUniform<float, CUDAContext>(
    const int N,
    const float low,
    const float high,
    float* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateUniform(ctx->curand_generator(), y, N));
  math::Scale(N, high - low, y, y, ctx);
  math::Bias(N, low, y, y, ctx);
}

template <>
DRAGON_API void RandomUniform<double, CUDAContext>(
    const int N,
    const float low,
    const float high,
    double* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateUniformDouble(ctx->curand_generator(), y, N));
  math::Scale(N, high - low, y, y, ctx);
  math::Bias(N, low, y, y, ctx);
}

template <>
DRAGON_API void RandomNormal<float, CUDAContext>(
    const int N,
    const float mu,
    const float sigma,
    float* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateNormal(ctx->curand_generator(), y, N, mu, sigma));
}

template <>
DRAGON_API void RandomNormal<double, CUDAContext>(
    const int N,
    const float mu,
    const float sigma,
    double* y,
    CUDAContext* ctx) {
  auto* rng = ctx->curand_generator();
  CURAND_CHECK(curandGenerateNormalDouble(rng, y, N, mu, sigma));
}

#define DEFINE_RANDOM_CAST_FUNC(name, T)                                     \
  template <>                                                                \
  DRAGON_API void name<T, CUDAContext>(                                      \
      const int N,                                                           \
      const float arg1,                                                      \
      const float arg2,                                                      \
      T* y,                                                                  \
      CUDAContext* ctx) {                                                    \
    auto* x = ctx->workspace()->data<float, CUDAContext>(N, "BufferKernel"); \
    math::name(N, arg1, arg2, x, ctx);                                       \
    math::Cast(N, x, y, ctx);                                                \
  }

DEFINE_RANDOM_CAST_FUNC(RandomUniform, float16);
DEFINE_RANDOM_CAST_FUNC(RandomUniform, bfloat16);
DEFINE_RANDOM_CAST_FUNC(RandomNormal, float16);
DEFINE_RANDOM_CAST_FUNC(RandomNormal, bfloat16);
#undef DEFINE_RANDOM_CAST_FUNC

} // namespace math

} // namespace dragon
