#include "dragon/utils/conversions.h"
#include "dragon/utils/math/elementwise.h"

namespace dragon {

namespace math {

namespace {

template <typename InputT, typename OutputT>
void _Cast(const int N, const InputT* x, OutputT* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = convert::To<OutputT>(x[i]);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_CAST_FUNC(InputT, OutputT)                          \
  template <>                                                      \
  DRAGON_API void Cast<InputT, OutputT, CPUContext>(               \
      const int N, const InputT* x, OutputT* y, CPUContext* ctx) { \
    _Cast(N, x, y);                                                \
  }

#define DEFINE_UNSUPPORTED_CAST_FUNC(InputT, OutputT)                    \
  template <>                                                            \
  DRAGON_API void Cast<InputT, OutputT, CPUContext>(                     \
      const int N, const InputT* x, OutputT* y, CPUContext* ctx) {       \
    LOG(FATAL) << "Unsupported conversion: "                             \
               << dtypes::to_string(TypeMeta::Make<InputT>()) << " -> "  \
               << dtypes::to_string(TypeMeta::Make<OutputT>());          \
  }                                                                      \
  template <>                                                            \
  DRAGON_API void Cast<OutputT, InputT, CPUContext>(                     \
      const int N, const OutputT* x, InputT* y, CPUContext* ctx) {       \
    LOG(FATAL) << "Unsupported conversion: "                             \
               << dtypes::to_string(TypeMeta::Make<OutputT>()) << " -> " \
               << dtypes::to_string(TypeMeta::Make<InputT>());           \
  }

#define DEFINE_CAST_FUNC_TO(T)  \
  DEFINE_CAST_FUNC(T, bool);    \
  DEFINE_CAST_FUNC(T, int8_t);  \
  DEFINE_CAST_FUNC(T, uint8_t); \
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
