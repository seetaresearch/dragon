#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/random.h"

namespace dragon {

namespace math {

template <>
DRAGON_API void RandomUniform<uint32_t, CUDAContext>(
    const int n,
    const float low,
    const float high,
    uint32_t* y,
    CUDAContext* ctx) {
  // Note that we ignore the low / high
  // CuRand could only generates in the range of [0, uint32]
  auto* rng = ctx->curand_generator();
  CURAND_CHECK(curandGenerate(rng, y, n));
}

template <>
DRAGON_API void RandomUniform<float16, CUDAContext>(
    const int n,
    const float low,
    const float high,
    float16* y,
    CUDAContext* ctx) {
  NOT_IMPLEMENTED;
}

template <>
DRAGON_API void RandomUniform<float, CUDAContext>(
    const int n,
    const float low,
    const float high,
    float* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateUniform(ctx->curand_generator(), y, n));
  Scale(n, high - low, y, y, ctx);
  Bias(n, low, y, y, ctx);
}

template <>
DRAGON_API void RandomUniform<double, CUDAContext>(
    const int n,
    const float low,
    const float high,
    double* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateUniformDouble(ctx->curand_generator(), y, n));
  Scale(n, high - low, y, y, ctx);
  Bias(n, low, y, y, ctx);
}

template <>
DRAGON_API void RandomNormal<float16, CUDAContext>(
    const int n,
    const float mu,
    const float sigma,
    float16* y,
    CUDAContext* ctx) {
  NOT_IMPLEMENTED;
}

template <>
DRAGON_API void RandomNormal<float, CUDAContext>(
    const int n,
    const float mu,
    const float sigma,
    float* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateNormal(ctx->curand_generator(), y, n, mu, sigma));
}

template <>
DRAGON_API void RandomNormal<double, CUDAContext>(
    const int n,
    const float mu,
    const float sigma,
    double* y,
    CUDAContext* ctx) {
  CURAND_CHECK(
      curandGenerateNormalDouble(ctx->curand_generator(), y, n, mu, sigma));
}

} // namespace math

} // namespace dragon

#endif // USE_CUDA
