#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _BiasAdd(
    const int outer_dim,
    const int axis_dim,
    const T* x,
    const T* b,
    T* y) {
  EigenArrayMap<T>(y, axis_dim, outer_dim) =
      ConstEigenArrayMap<T>(x, axis_dim, outer_dim).colwise() +
      ConstEigenVectorArrayMap<T>(b, axis_dim);
}

template <typename T>
void _BiasAdd(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    const T* b,
    T* y) {
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < axis_dim; ++j) {
      EigenVectorArrayMap<T>(y, inner_dim) =
          ConstEigenVectorArrayMap<T>(x, inner_dim) + b[j];
      x += inner_dim;
      y += inner_dim;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void BiasAdd<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const float16* x,
    const float16* b,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                        \
  template <>                                            \
  void BiasAdd<T, CPUContext>(                           \
      const int outer_dim,                               \
      const int inner_dim,                               \
      const int axis_dim,                                \
      const T* x,                                        \
      const T* b,                                        \
      T* y,                                              \
      CPUContext* ctx) {                                 \
    if (inner_dim == 1) {                                \
      _BiasAdd(outer_dim, axis_dim, x, b, y);            \
    } else {                                             \
      _BiasAdd(outer_dim, inner_dim, axis_dim, x, b, y); \
    }                                                    \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
