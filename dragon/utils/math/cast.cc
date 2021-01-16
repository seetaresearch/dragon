#include "dragon/utils/conversions.h"
#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/math/elementwise.h"

namespace dragon {

namespace math {

namespace {

template <typename InputT, typename OutputT>
void _Cast(const int n, const InputT* x, OutputT* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(n))
#endif
  for (int i = 0; i < n; ++i) {
    y[i] = convert::To<OutputT>(x[i]);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_CAST_KERNEL_LAUNCHER(InputT, OutputT)               \
  template <>                                                      \
  void Cast<InputT, OutputT, CPUContext>(                          \
      const int n, const InputT* x, OutputT* y, CPUContext* ctx) { \
    _Cast(n, x, y);                                                \
  }

#define DEFINE_UNSUPPORTED_KERNEL_LAUNCHER(InputT, OutputT)             \
  template <>                                                           \
  void Cast<InputT, OutputT, CPUContext>(                               \
      const int n, const InputT* x, OutputT* y, CPUContext* ctx) {      \
    LOG(FATAL) << "Unsupported conversion: "                            \
               << types::to_string(TypeMeta::Make<InputT>()) << " -> "  \
               << types::to_string(TypeMeta::Make<OutputT>());          \
  }                                                                     \
  template <>                                                           \
  void Cast<OutputT, InputT, CPUContext>(                               \
      const int n, const OutputT* x, InputT* y, CPUContext* ctx) {      \
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
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
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
