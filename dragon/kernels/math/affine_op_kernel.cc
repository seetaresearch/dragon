#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _Affine(
    const int outer_dim,
    const int axis_dim,
    const T* x,
    const T* w,
    const T* b,
    T* y) {
  if (b != nullptr) {
    EigenArrayMap<T>(y, axis_dim, outer_dim) =
        (ConstEigenArrayMap<T>(x, axis_dim, outer_dim).colwise() *
         ConstEigenVectorArrayMap<T>(w, axis_dim))
            .colwise() +
        ConstEigenVectorArrayMap<T>(b, axis_dim);
  } else {
    EigenArrayMap<T>(y, axis_dim, outer_dim) =
        ConstEigenArrayMap<T>(x, axis_dim, outer_dim).colwise() *
        ConstEigenVectorArrayMap<T>(w, axis_dim);
  }
}

template <typename T>
void _Affine(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* x,
    const T* w,
    const T* b,
    T* y) {
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < axis_dim; ++j) {
      if (b != nullptr) {
        EigenVectorArrayMap<T>(y, inner_dim) =
            ConstEigenVectorArrayMap<T>(x, inner_dim) * w[j] + b[j];
      } else {
        EigenVectorArrayMap<T>(y, inner_dim) =
            ConstEigenVectorArrayMap<T>(x, inner_dim) * w[j];
      }
      x += inner_dim;
      y += inner_dim;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Affine<float16, CPUContext>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float16* x,
    const float16* w,
    const float16* b,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void Affine<T, CPUContext>(                              \
      const int outer_dim,                                 \
      const int axis_dim,                                  \
      const int inner_dim,                                 \
      const T* x,                                          \
      const T* w,                                          \
      const T* b,                                          \
      T* y,                                                \
      CPUContext* ctx) {                                   \
    if (inner_dim == 1) {                                  \
      _Affine(outer_dim, axis_dim, x, w, b, y);            \
    } else {                                               \
      _Affine(outer_dim, axis_dim, inner_dim, x, w, b, y); \
    }                                                      \
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
