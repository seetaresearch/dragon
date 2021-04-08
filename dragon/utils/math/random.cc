#include "dragon/utils/math/random.h"

namespace dragon {

namespace math {

template <>
DRAGON_API void RandomUniform<float16, CPUContext>(
    const int N,
    const float low,
    const float high,
    float16* x,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
DRAGON_API void RandomNormal<float16, CPUContext>(
    const int N,
    const float mu,
    const float sigma,
    float16* x,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
DRAGON_API void TruncatedNormal<float16, CPUContext>(
    const int N,
    const float mu,
    const float sigma,
    const float low,
    const float high,
    float16* x,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

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

#define DEFINE_RANDOM_UNIFORM_FUNC(T, key)                                     \
  template <>                                                                  \
  DRAGON_API void RandomUniform<T, CPUContext>(                                \
      const int N, const float low, const float high, T* y, CPUContext* ctx) { \
    std::uniform_##key##_distribution<T> distribution(low, high);              \
    auto* rng = ctx->rand_generator();                                         \
    for (int i = 0; i < N; ++i) {                                              \
      y[i] = distribution(*rng);                                               \
    }                                                                          \
  }

DEFINE_RANDOM_UNIFORM_FUNC(float, real);
DEFINE_RANDOM_UNIFORM_FUNC(double, real);
#undef DEFINE_RANDOM_UNIFORM_FUNC

#define DEFINE_RANDOM_NORMAL_FUNC(T)                                           \
  template <>                                                                  \
  DRAGON_API void RandomNormal<T, CPUContext>(                                 \
      const int N, const float mu, const float sigma, T* y, CPUContext* ctx) { \
    std::normal_distribution<T> distribution(mu, sigma);                       \
    auto* rng = ctx->rand_generator();                                         \
    for (int i = 0; i < N; ++i)                                                \
      y[i] = distribution(*rng);                                               \
  }

DEFINE_RANDOM_NORMAL_FUNC(float);
DEFINE_RANDOM_NORMAL_FUNC(double);
#undef DEFINE_RANDOM_NORMAL_FUNC

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

} // namespace math

} // namespace dragon
