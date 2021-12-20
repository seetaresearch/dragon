#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
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
  Scale(N, high - low, y, y, ctx);
  Bias(N, low, y, y, ctx);
}

template <>
DRAGON_API void RandomUniform<double, CUDAContext>(
    const int N,
    const float low,
    const float high,
    double* y,
    CUDAContext* ctx) {
  CURAND_CHECK(curandGenerateUniformDouble(ctx->curand_generator(), y, N));
  Scale(N, high - low, y, y, ctx);
  Bias(N, low, y, y, ctx);
}

template <>
DRAGON_API void RandomUniform<float16, CUDAContext>(
    const int N,
    const float low,
    const float high,
    float16* y,
    CUDAContext* ctx) {
  auto* scratch = ctx->workspace()->template data<CUDAContext>(
      sizeof(float) * N, "BufferKernel");
  RandomUniform(N, low, high, (float*)scratch, ctx);
  Cast(N, (float*)scratch, y, ctx);
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

template <>
DRAGON_API void RandomNormal<float16, CUDAContext>(
    const int N,
    const float mu,
    const float sigma,
    float16* y,
    CUDAContext* ctx) {
  auto* scratch = ctx->workspace()->template data<CUDAContext>(
      sizeof(float) * N, "BufferKernel");
  RandomNormal(N, mu, sigma, (float*)scratch, ctx);
  Cast(N, (float*)scratch, y, ctx);
}

} // namespace math

} // namespace dragon

#endif // USE_CUDA
