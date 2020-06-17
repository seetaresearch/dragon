#include "dragon/utils/cast.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename Tx, typename Ty>
void _Cast(const int count, const Tx* x, Ty* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    y[i] = cast::to<Ty>(x[i]);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, Ty)                \
  template <>                                                 \
  void Cast<Tx, Ty, CPUContext>(                              \
      const int count, const Tx* x, Ty* y, CPUContext* ctx) { \
    _Cast(count, x, y);                                       \
  }

#define DEFINE_FP16_KERNEL_LAUNCHER(T)                                         \
  template <>                                                                  \
  void Cast<float16, T, CPUContext>(                                           \
      const int count, const float16* x, T* y, CPUContext* ctx) {              \
    LOG(FATAL) << "Not Implemented: float16 -> "                               \
               << types::to_string(TypeMeta::Make<T>());                       \
  }                                                                            \
  template <>                                                                  \
  void Cast<T, float16, CPUContext>(                                           \
      const int count, const T* x, float16* y, CPUContext* ctx) {              \
    LOG(FATAL) << "Not Implemented: " << types::to_string(TypeMeta::Make<T>()) \
               << " -> float16";                                               \
  }

#define DEFINE_KERNEL_LAUNCHER(Tx)             \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, bool);    \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, int8_t);  \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, uint8_t); \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, int);     \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, int64_t); \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, float);   \
  DEFINE_GENERIC_KERNEL_LAUNCHER(Tx, double);

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GENERIC_KERNEL_LAUNCHER(float16, float);
DEFINE_GENERIC_KERNEL_LAUNCHER(float, float16);
DEFINE_GENERIC_KERNEL_LAUNCHER(float16, float16);

DEFINE_FP16_KERNEL_LAUNCHER(bool);
DEFINE_FP16_KERNEL_LAUNCHER(uint8_t);
DEFINE_FP16_KERNEL_LAUNCHER(int8_t);
DEFINE_FP16_KERNEL_LAUNCHER(int);
DEFINE_FP16_KERNEL_LAUNCHER(int64_t);
DEFINE_FP16_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GENERIC_KERNEL_LAUNCHER
#undef DEFINE_FP16_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
