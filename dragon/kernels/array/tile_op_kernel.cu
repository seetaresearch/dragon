#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, int D>
__global__ void _Tile(
    const int N,
    const int num_dims,
    const SimpleArray<int, D> X_dims,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += (r % X_dims.data[d]) * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, typename AccT>
__global__ void _TileGrad(
    const int NxC1xS,
    const int C1xS,
    const int C2xS,
    const T* dy,
    T* dx) {
  const int repeats = C2xS / C1xS;
  CUDA_1D_KERNEL_LOOP(xi, NxC1xS) {
    const int i = xi / C1xS;
    const int j = xi % C1xS;
    const T* offset_dy = dy + i * C2xS + j;
    AccT val = AccT(0);
    for (int k = 0; k < repeats; ++k) {
      val += convert::To<AccT>(*offset_dy);
      offset_dy += C1xS;
    }
    dx[xi] = convert::To<T>(val);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                     \
  template <>                                                         \
  void Tile<T, CUDAContext>(                                          \
      const int num_dims,                                             \
      const int64_t* x_dims,                                          \
      const int64_t* x_strides,                                       \
      const int64_t* y_dims,                                          \
      const T* x,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    CUDA_TENSOR_DIMS_CHECK(num_dims);                                 \
    SimpleArray<int, CUDA_TENSOR_MAX_DIMS> X_dims, X_strides, Y_dims; \
    const auto N = std::accumulate(                                   \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());    \
    for (int i = 0; i < num_dims; ++i) {                              \
      X_dims.data[i] = x_dims[i];                                     \
      X_strides.data[i] = x_strides[i];                               \
      Y_dims.data[i] = y_dims[i];                                     \
    }                                                                 \
    _Tile<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        N, num_dims, X_dims, X_strides, Y_dims, x, y);                \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void TileGrad<T, CUDAContext>(                                        \
      const int N,                                                      \
      const int CxS,                                                    \
      const int repeats,                                                \
      const T* dy,                                                      \
      T* dx,                                                            \
      CUDAContext* ctx) {                                               \
    const auto NxCxS = N * CxS;                                         \
    _TileGrad<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(  \
            NxCxS,                                                      \
            CxS,                                                        \
            CxS * repeats,                                              \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),     \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));          \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
