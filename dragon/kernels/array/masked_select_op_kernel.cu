#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/conversions.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename IndexType, typename ValueType>
__global__ void _MaskedSelect(
    const int nthreads,
    const IndexType* index,
    const ValueType* x,
    ValueType* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = x[index[i]];
  }
}

template <typename IndexType, typename ValueType>
__global__ void _MaskedSelectGrad(
    const int nthreads,
    const IndexType* index,
    const ValueType* dy,
    ValueType* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    dx[index[i]] = dy[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(IndexType, ValueType)      \
  template <>                                             \
  void MaskedSelect<IndexType, ValueType, CUDAContext>(   \
      const int num_selected,                             \
      const IndexType* index,                             \
      const ValueType* x,                                 \
      ValueType* y,                                       \
      CUDAContext* ctx) {                                 \
    _MaskedSelect<<<                                      \
        CUDA_BLOCKS(num_selected),                        \
        CUDA_THREADS,                                     \
        0,                                                \
        ctx->cuda_stream()>>>(num_selected, index, x, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(IndexType, ValueType)   \
  template <>                                               \
  void MaskedSelectGrad<IndexType, ValueType, CUDAContext>( \
      const int count,                                      \
      const int num_selected,                               \
      const IndexType* index,                               \
      const ValueType* dy,                                  \
      ValueType* dx,                                        \
      CUDAContext* ctx) {                                   \
    math::Set(count, convert::To<ValueType>(0.f), dx, ctx); \
    _MaskedSelectGrad<<<                                    \
        CUDA_BLOCKS(num_selected),                          \
        CUDA_THREADS,                                       \
        0,                                                  \
        ctx->cuda_stream()>>>(num_selected, index, dy, dx); \
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

#endif // USE_CUDA
