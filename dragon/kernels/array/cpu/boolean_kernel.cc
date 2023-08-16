#include "dragon/kernels/array/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename IndexT, typename ValueT>
void _BooleanMask(
    const int N,
    const IndexT* index,
    const ValueT* x,
    ValueT* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = x[index[i]];
  }
}

template <typename IndexT, typename ValueT>
void _BooleanMaskGrad(
    const int N,
    const IndexT* index,
    const ValueT* dy,
    ValueT* dx) {
  for (int i = 0; i < N; ++i) {
    dx[index[i]] = dy[i];
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(IndexT, ValueT)  \
  template <>                                   \
  void BooleanMask<IndexT, ValueT, CPUContext>( \
      const int N,                              \
      const IndexT* index,                      \
      const ValueT* x,                          \
      ValueT* y,                                \
      CPUContext* ctx) {                        \
    _BooleanMask(N, index, x, y);               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(IndexT, ValueT) \
  template <>                                       \
  void BooleanMaskGrad<IndexT, ValueT, CPUContext>( \
      const int N,                                  \
      const IndexT* index,                          \
      const ValueT* dy,                             \
      ValueT* dx,                                   \
      CPUContext* ctx) {                            \
    _BooleanMaskGrad(N, index, dy, dx);             \
  }

DEFINE_KERNEL_LAUNCHER(int, bool);
DEFINE_KERNEL_LAUNCHER(int, uint8_t);
DEFINE_KERNEL_LAUNCHER(int, int8_t);
DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int, int64_t);
DEFINE_KERNEL_LAUNCHER(int, float16);
DEFINE_KERNEL_LAUNCHER(int, bfloat16);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int, double);
DEFINE_KERNEL_LAUNCHER(int64_t, bool);
DEFINE_KERNEL_LAUNCHER(int64_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(int64_t, int8_t);
DEFINE_KERNEL_LAUNCHER(int64_t, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_KERNEL_LAUNCHER(int64_t, bfloat16);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_GRAD_KERNEL_LAUNCHER(int, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(int, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(int, float);
DEFINE_GRAD_KERNEL_LAUNCHER(int, double);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, float);
DEFINE_GRAD_KERNEL_LAUNCHER(int64_t, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
