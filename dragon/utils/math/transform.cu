#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/transform.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename T>
__global__ void _AffineChannel(
    const int NxC,
    const int C,
    const T* x,
    const T* scale,
    const T* bias,
    T* y) {
  auto op3 = math::FMAFunctor<T>();
  auto op2 = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, NxC) {
    if (bias != nullptr) {
      y[i] = op3(x[i], __ldg(scale + i % C), __ldg(bias + i % C));
    } else {
      y[i] = op2(x[i], __ldg(scale + i % C));
    }
  }
}

template <typename T>
__global__ void _AffineChannel(
    const int NxCxS,
    const int C,
    const int S,
    const T* x,
    const T* scale,
    const T* bias,
    T* y) {
  auto op3 = math::FMAFunctor<T>();
  auto op2 = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = (i / S) % C;
    if (bias != nullptr) {
      y[i] = op3(x[i], __ldg(scale + j), __ldg(bias + j));
    } else {
      y[i] = op2(x[i], __ldg(scale + j));
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
    T* y,
    CUDAContext* ctx) {
  const auto N = math::utils::Prod(num_dims, dims);
  if (num_dims == 1 && num_axes == 1 && axes[0] == 0) {
    _AffineChannel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, dims[0], x, scale, bias, y); // [NxC]
  } else if (num_dims == 2 && num_axes == 1 && axes[0] == 1) {
    _AffineChannel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, dims[1], x, scale, bias, y); // [N, C]
  } else if (num_dims == 2 && num_axes == 1 && axes[0] == 0) {
    _AffineChannel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, dims[0], dims[1], x, scale, bias, y); // [NxC, S]
  } else if (num_dims == 3 && num_axes == 1 && axes[0] == 1) {
    _AffineChannel<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, dims[1], dims[2], x, scale, bias, y); // [N, C, S]
  } else {
    LOG(FATAL) << "Unsupported affine dimensions.";
  }
}

} // namespace

#define DEFINE_AFFINE_FUNC(T)                                      \
  template <>                                                      \
  void Affine<T, CUDAContext>(                                     \
      const int num_dims,                                          \
      const int64_t* dims,                                         \
      const int num_axes,                                          \
      const int64_t* axes,                                         \
      const T* x,                                                  \
      const T* scale,                                              \
      const T* bias,                                               \
      T* y,                                                        \
      CUDAContext* ctx) {                                          \
    vec64_t new_dims, new_axes;                                    \
    math::utils::CollapseReduceAxes(                               \
        num_dims, dims, num_axes, axes, new_dims, new_axes);       \
    _AffineImpl(                                                   \
        new_dims.size(),                                           \
        new_dims.data(),                                           \
        new_axes.size(),                                           \
        new_axes.data(),                                           \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),     \
        reinterpret_cast<const math::ScalarType<T>::type*>(scale), \
        reinterpret_cast<const math::ScalarType<T>::type*>(bias),  \
        reinterpret_cast<math::ScalarType<T>::type*>(y),           \
        ctx);                                                      \
  }

DEFINE_AFFINE_FUNC(float);
DEFINE_AFFINE_FUNC(float16);
DEFINE_AFFINE_FUNC(double);
#undef DEFINE_AFFINE_FUNC

} // namespace math

} // namespace dragon

#endif // USE_CUDA
