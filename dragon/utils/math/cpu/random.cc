#include "dragon/utils/math/random.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math/elementwise.h"

namespace dragon {

namespace math {

#define DEFINE_RANDOM_FUNC(T)                                                 \
  template <>                                                                 \
  DRAGON_API void Random<T, CPUContext>(const int N, T* y, CPUContext* ctx) { \
    auto* rng = ctx->rand_generator();                                        \
    for (int i = 0; i < N; ++i) {                                             \
      y[i] = static_cast<T>((*rng)());                                        \
    }                                                                         \
  }

DEFINE_RANDOM_FUNC(uint32_t);
#undef DEFINE_RANDOM_FUNC

#define DEFINE_RANDOM_FUNC(name, T, Dist) \
  template <>                             \
  DRAGON_API void name<T, CPUContext>(    \
      const int N,                        \
      const float arg1,                   \
      const float arg2,                   \
      T* y,                               \
      CPUContext* ctx) {                  \
    Dist<T> distribution(arg1, arg2);     \
    auto* rng = ctx->rand_generator();    \
    for (int i = 0; i < N; ++i) {         \
      y[i] = distribution(*rng);          \
    }                                     \
  }

DEFINE_RANDOM_FUNC(RandomUniform, float, std::uniform_real_distribution);
DEFINE_RANDOM_FUNC(RandomUniform, double, std::uniform_real_distribution);
DEFINE_RANDOM_FUNC(RandomNormal, float, std::normal_distribution);
DEFINE_RANDOM_FUNC(RandomNormal, double, std::normal_distribution);
#undef DEFINE_RANDOM_FUNC

#define DEFINE_RANDOM_CAST_FUNC(name, T)                                    \
  template <>                                                               \
  DRAGON_API void name<T, CPUContext>(                                      \
      const int N,                                                          \
      const float arg1,                                                     \
      const float arg2,                                                     \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    auto* x = ctx->workspace()->data<float, CPUContext>(N, "BufferKernel"); \
    math::name(N, arg1, arg2, x, ctx);                                      \
    math::Cast(N, x, y, ctx);                                               \
  }

DEFINE_RANDOM_CAST_FUNC(RandomUniform, float16);
DEFINE_RANDOM_CAST_FUNC(RandomUniform, bfloat16);
DEFINE_RANDOM_CAST_FUNC(RandomNormal, float16);
DEFINE_RANDOM_CAST_FUNC(RandomNormal, bfloat16);
#undef DEFINE_RANDOM_CAST_FUNC

#define DEFINE_RANDOM_BERNOULI_FUNC(T)                     \
  template <>                                              \
  DRAGON_API void RandomBernoulli<T, CPUContext>(          \
      const int N, const float p, T* y, CPUContext* ctx) { \
    std::bernoulli_distribution distribution(p);           \
    auto* rng = ctx->rand_generator();                     \
    for (int i = 0; i < N; ++i)                            \
      y[i] = distribution(*rng);                           \
  }

DEFINE_RANDOM_BERNOULI_FUNC(uint8_t);
DEFINE_RANDOM_BERNOULI_FUNC(uint32_t);
DEFINE_RANDOM_BERNOULI_FUNC(float);
#undef DEFINE_RANDOM_BERNOULI_FUNC

#define DEFINE_TRUNCATED_NORMAL_FUNC(T)                  \
  template <>                                            \
  DRAGON_API void TruncatedNormal<T, CPUContext>(        \
      const int N,                                       \
      const float mu,                                    \
      const float sigma,                                 \
      const float low,                                   \
      const float high,                                  \
      T* y,                                              \
      CPUContext* ctx) {                                 \
    std::normal_distribution<T> distribution(mu, sigma); \
    auto* rng = ctx->rand_generator();                   \
    int cur_pos = 0;                                     \
    T gen_value;                                         \
    while (1) {                                          \
      gen_value = distribution(*rng);                    \
      if (gen_value < low) continue;                     \
      if (gen_value > high) continue;                    \
      y[cur_pos++] = gen_value;                          \
      if (cur_pos >= N) break;                           \
    }                                                    \
  }

DEFINE_TRUNCATED_NORMAL_FUNC(float);
DEFINE_TRUNCATED_NORMAL_FUNC(double);
#undef DEFINE_TRUNCATED_NORMAL_FUNC

#define DEFINE_TRUNCATED_NORMAL_CAST_FUNC(T)                                \
  template <>                                                               \
  DRAGON_API void TruncatedNormal<T, CPUContext>(                           \
      const int N,                                                          \
      const float mu,                                                       \
      const float sigma,                                                    \
      const float low,                                                      \
      const float high,                                                     \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    auto* x = ctx->workspace()->data<float, CPUContext>(N, "BufferKernel"); \
    math::TruncatedNormal(N, mu, sigma, low, high, x, ctx);                 \
    math::Cast(N, x, y, ctx);                                               \
  }

DEFINE_TRUNCATED_NORMAL_CAST_FUNC(float16);
DEFINE_TRUNCATED_NORMAL_CAST_FUNC(bfloat16);
#undef DEFINE_TRUNCATED_NORMAL_CAST_FUNC

} // namespace math

} // namespace dragon
