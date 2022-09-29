#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename T, size_t L>
void _AlignedTranspose(
    const int num_dims,
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y) {
  const auto N = utils::Prod(num_dims, y_dims);
  using ScalarT = typename std::aligned_storage<L, L>::type;
  const auto* x_aligned = reinterpret_cast<const ScalarT*>(x);
  auto* y_aligned = reinterpret_cast<ScalarT*>(y);
  vec64_t index(num_dims, 0);
  for (int yi = 0; yi < N; ++yi) {
    int64_t xi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      xi += index[d] * x_strides[d];
    }
    y_aligned[yi] = x_aligned[xi];
    utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

template <typename T>
void _TransposeImpl(
    const int num_dims,
    const vec64_t& dims,
    const vec64_t& axes,
    const T* x,
    T* y) {
  auto aligned_size = sizeof(T);
  if (axes.back() == num_dims - 1) {
    aligned_size = utils::GetAlignedSize<T, 16>(dims.back());
  }
  vec64_t X_dims(num_dims), X_strides(num_dims), Y_dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    X_dims[i] = dims[i];
  }
  X_dims[num_dims - 1] /= int64_t(aligned_size / sizeof(T));
  utils::ComputeTransposeStrides(
      num_dims, X_dims.data(), axes.data(), X_strides.data());
  for (int i = 0; i < num_dims; ++i) {
    Y_dims[i] = X_dims[axes[i]];
  }
  if (aligned_size == 1) {
    _AlignedTranspose<T, 1>(num_dims, X_strides.data(), Y_dims.data(), x, y);
  } else if (aligned_size == 2) {
    _AlignedTranspose<T, 2>(num_dims, X_strides.data(), Y_dims.data(), x, y);
  } else if (aligned_size == 4) {
    _AlignedTranspose<T, 4>(num_dims, X_strides.data(), Y_dims.data(), x, y);
  } else if (aligned_size == 8) {
    _AlignedTranspose<T, 8>(num_dims, X_strides.data(), Y_dims.data(), x, y);
  } else if (aligned_size == 16) {
    _AlignedTranspose<T, 16>(num_dims, X_strides.data(), Y_dims.data(), x, y);
  } else {
    LOG(FATAL) << "Unsupported aligned size: " << aligned_size;
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
    _TransposeImpl(new_dims.size(), new_dims, new_axes, x, y);              \
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
