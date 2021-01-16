#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#else
#define LDG(x, i) x[i]
#endif

template <typename T>
__global__ void
_BiasAdd(const int nthreads, const int axis_dim, const T* x, const T* b, T* y) {
  auto Plus = math::PlusFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = Plus(x[i], LDG(b, i % axis_dim));
  }
}

template <typename T>
__global__ void _BiasAdd(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const T* x,
    const T* b,
    T* y) {
  auto Plus = math::PlusFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = Plus(x[i], LDG(b, (i / inner_dim) % axis_dim));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                \
  template <>                                                    \
  void BiasAdd<T, CUDAContext>(                                  \
      const int outer_dim,                                       \
      const int inner_dim,                                       \
      const int axis_dim,                                        \
      const T* x,                                                \
      const T* b,                                                \
      T* y,                                                      \
      CUDAContext* ctx) {                                        \
    const auto nthreads = outer_dim * axis_dim * inner_dim;      \
    if (inner_dim == 1) {                                        \
      _BiasAdd<<<                                                \
          CUDA_BLOCKS(nthreads),                                 \
          CUDA_THREADS,                                          \
          0,                                                     \
          ctx->cuda_stream()>>>(                                 \
          nthreads,                                              \
          axis_dim,                                              \
          reinterpret_cast<const math::ScalarType<T>::type*>(x), \
          reinterpret_cast<const math::ScalarType<T>::type*>(b), \
          reinterpret_cast<math::ScalarType<T>::type*>(y));      \
    } else {                                                     \
      _BiasAdd<<<                                                \
          CUDA_BLOCKS(nthreads),                                 \
          CUDA_THREADS,                                          \
          0,                                                     \
          ctx->cuda_stream()>>>(                                 \
          nthreads,                                              \
          inner_dim,                                             \
          axis_dim,                                              \
          reinterpret_cast<const math::ScalarType<T>::type*>(x), \
          reinterpret_cast<const math::ScalarType<T>::type*>(b), \
          reinterpret_cast<math::ScalarType<T>::type*>(y));      \
    }                                                            \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
