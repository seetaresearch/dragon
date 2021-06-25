#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
void _Transpose(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto N =
      std::accumulate(y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());
  vec64_t index(num_dims, 0);
  for (int yi = 0; yi < N; ++yi) {
    int64_t xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += index[d] * x_strides[d];
    }
    y[yi] = x[xi];
    utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

} // namespace

#define DEFINE_TRANSPOSE_FUNC(T)                                            \
  template <>                                                               \
  DRAGON_API void Transpose<T, CPUContext>(                                 \
      const int num_dims,                                                   \
      const int64_t* dims,                                                  \
      const int64_t* axes,                                                  \
      const T* x,                                                           \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    vec64_t new_dims, new_axes;                                             \
    utils::CollapseTransposeAxes(num_dims, dims, axes, new_dims, new_axes); \
    const int num_axes = new_dims.size();                                   \
    vec64_t X_strides(num_axes), Y_dims(num_axes);                          \
    utils::ComputeTransposeStrides(                                         \
        num_axes, new_dims.data(), new_axes.data(), X_strides.data());      \
    for (int i = 0; i < num_axes; ++i) {                                    \
      Y_dims[i] = new_dims[new_axes[i]];                                    \
    }                                                                       \
    _Transpose(num_axes, X_strides.data(), Y_dims.data(), x, y);            \
  }

DEFINE_TRANSPOSE_FUNC(bool);
DEFINE_TRANSPOSE_FUNC(uint8_t);
DEFINE_TRANSPOSE_FUNC(int8_t);
DEFINE_TRANSPOSE_FUNC(int);
DEFINE_TRANSPOSE_FUNC(int64_t);
DEFINE_TRANSPOSE_FUNC(float16);
DEFINE_TRANSPOSE_FUNC(float);
DEFINE_TRANSPOSE_FUNC(double);
#undef DEFINE_TRANSPOSE_FUNC

} // namespace math

} // namespace dragon
