#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename IndexType, typename ValueType>
void _MaskedSelect(
    const int num_selected,
    const IndexType* index,
    const ValueType* x,
    ValueType* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(num_selected))
#endif
  for (int i = 0; i < num_selected; ++i) {
    y[i] = x[index[i]];
  }
}

template <typename IndexType, typename ValueType>
void _MaskedSelectGrad(
    const int num_selected,
    const IndexType* index,
    const ValueType* dy,
    ValueType* dx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(num_selected))
#endif
  for (int i = 0; i < num_selected; ++i) {
    dx[index[i]] = dy[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(IndexType, ValueType)   \
  template <>                                          \
  void MaskedSelect<IndexType, ValueType, CPUContext>( \
      const int num_selected,                          \
      const IndexType* index,                          \
      const ValueType* x,                              \
      ValueType* y,                                    \
      CPUContext* ctx) {                               \
    _MaskedSelect(num_selected, index, x, y);          \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(IndexType, ValueType)   \
  template <>                                               \
  void MaskedSelectGrad<IndexType, ValueType, CPUContext>(  \
      const int count,                                      \
      const int num_selected,                               \
      const IndexType* index,                               \
      const ValueType* dy,                                  \
      ValueType* dx,                                        \
      CPUContext* ctx) {                                    \
    math::Set(count, convert::To<ValueType>(0.f), dx, ctx); \
    _MaskedSelectGrad(num_selected, index, dy, dx);         \
  }

DEFINE_KERNEL_LAUNCHER(int, bool);
DEFINE_KERNEL_LAUNCHER(int, int8_t);
DEFINE_KERNEL_LAUNCHER(int, uint8_t);
DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int, int64_t);
DEFINE_KERNEL_LAUNCHER(int, float16);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int, double);
DEFINE_KERNEL_LAUNCHER(int64_t, bool);
DEFINE_KERNEL_LAUNCHER(int64_t, int8_t);
DEFINE_KERNEL_LAUNCHER(int64_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(int64_t, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);

DEFINE_GRAD_KERNEL_LAUNCHER(int, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(int, float);
DEFINE_GRAD_KERNEL_LAUNCHER(int, double);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, float);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
