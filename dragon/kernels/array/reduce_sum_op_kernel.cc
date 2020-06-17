#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

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
  const auto count =
      std::accumulate(x_dims, x_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  int64_t yi;
  for (int xi = 0; xi < count; ++xi) {
    yi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      yi += (index[d] % y_dims[d]) * y_strides[d];
    }
    dx[xi] = dy[yi] * scale;
    utils::math::IncreaseIndexInDims(num_dims, x_dims, index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ReduceSumGrad<float16, CPUContext>(
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* y_dims,
    const int64_t* y_strides,
    const float scale,
    const float16* dy,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // ReduceSumGrad

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

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
