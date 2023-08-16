#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

#define DEFINE_KERNEL_LAUNCHER(T)                        \
  template <>                                            \
  void Clip<T, CPUContext>(                              \
      const int N,                                       \
      const float low,                                   \
      const float high,                                  \
      const T* x,                                        \
      T* y,                                              \
      CPUContext* ctx) {                                 \
    using EigenT = typename math::Traits<T>::eigen_type; \
    EigenVectorMap<EigenT> Y((EigenT*)y, N);             \
    ConstEigenVectorMap<EigenT> X((const EigenT*)x, N);  \
    Y = X.cwiseMax(EigenT(low)).cwiseMin(EigenT(high));  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void ClipGrad<T, CPUContext>(                                       \
      const int N,                                                    \
      const float low,                                                \
      const float high,                                               \
      const T* dy,                                                    \
      const T* x,                                                     \
      T* dx,                                                          \
      CPUContext* ctx) {                                              \
    using EigenT = typename math::Traits<T>::eigen_type;              \
    EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);                   \
    ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);          \
    ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);        \
    dX = (X < EigenT(low) || X > EigenT(high)).select(EigenT(0), dY); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
