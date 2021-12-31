#include "dragon/utils/math/transform.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math/reduce.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
void _AffineChannel(
    const int N,
    const int C,
    const T* x,
    const T* scale,
    const T* bias,
    T* y) {
  EigenArrayMap<T> Y(y, C, N);
  ConstEigenArrayMap<T> X(x, C, N);
  Y = X.colwise() * ConstEigenVectorArrayMap<T>(scale, C);
  if (bias != nullptr) {
    Y.colwise() += ConstEigenVectorArrayMap<T>(bias, C);
  }
}

template <typename T>
void _AffineChannel(
    const int N,
    const int C,
    const int S,
    const T* x,
    const T* scale,
    const T* bias,
    T* y) {
  const auto CxS = C * S;
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<T> Y(y + i * CxS, S, C);
    ConstEigenArrayMap<T> X(x + i * CxS, S, C);
    Y = X.rowwise() * ConstEigenVectorArrayMap<T>(scale, C).transpose();
    if (bias != nullptr) {
      Y.rowwise() += ConstEigenVectorArrayMap<T>(bias, C).transpose();
    }
  }
}

template <typename T>
void _AffineImpl(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const T* x,
    const T* scale,
    const T* bias,
    T* y) {
  if (num_dims == 2 && num_axes == 1 && axes[0] == 1) {
    _AffineChannel(dims[0], dims[1], x, scale, bias, y);
  } else if (num_dims == 3 && num_axes == 1 && axes[0] == 1) {
    _AffineChannel(dims[0], dims[1], dims[2], x, scale, bias, y);
  } else {
    LOG(FATAL) << "Unsupported affine dimensions.";
  }
}

} // namespace

template <>
void Affine<float16, CPUContext>(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const float16* x,
    const float16* scale,
    const float16* bias,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_AFFINE_FUNC(T)                                \
  template <>                                                \
  void Affine<T, CPUContext>(                                \
      const int num_dims,                                    \
      const int64_t* dims,                                   \
      const int num_axes,                                    \
      const int64_t* axes,                                   \
      const T* x,                                            \
      const T* scale,                                        \
      const T* bias,                                         \
      T* y,                                                  \
      CPUContext* ctx) {                                     \
    vec64_t new_dims, new_axes;                              \
    math::utils::CollapseReduceAxes(                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes); \
    _AffineImpl(                                             \
        new_dims.size(),                                     \
        new_dims.data(),                                     \
        new_axes.size(),                                     \
        new_axes.data(),                                     \
        x,                                                   \
        scale,                                               \
        bias,                                                \
        y);                                                  \
  }

DEFINE_AFFINE_FUNC(float);
DEFINE_AFFINE_FUNC(double);
#undef DEFINE_AFFINE_FUNC

} // namespace math

} // namespace dragon
