#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ReduceSumGrad(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_dims,
    const int64_t* y_strides,
    const float scale,
    const T* dy,
    T* dx) {
  const auto N = math::utils::Prod(num_dims, x_dims);
  vec64_t index(num_dims, 0);
  int64_t yi;
  for (int xi = 0; xi < N; ++xi) {
    yi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      yi += (index[d] % y_dims[d]) * y_strides[d];
    }
    dx[xi] = convert::To<T>(convert::To<float>(dy[yi]) * scale);
    math::utils::IncreaseIndexInDims(num_dims, x_dims, index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void ReduceSumGrad<T, CPUContext>(                                    \
      const int num_dims,                                               \
      const int64_t* x_dims,                                            \
      const int64_t* y_dims,                                            \
      const int64_t* y_strides,                                         \
      const float scale,                                                \
      const T* dy,                                                      \
      T* dx,                                                            \
      CPUContext* ctx) {                                                \
    _ReduceSumGrad(num_dims, x_dims, y_dims, y_strides, scale, dy, dx); \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
