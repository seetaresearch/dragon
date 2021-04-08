#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _CosGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = -dy[i] * sin(x[i]);
  }
}

template <>
__global__ void
_CosGrad<half>(const int N, const half* dy, const half* x, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __float2half(-__half2float(dy[i]) * sin(__half2float(x[i])));
  }
} // CosGrad

template <>
__global__ void
_CosGrad<half2>(const int N, const half2* dy, const half2* x, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(x[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(-grad.x * sin(val.x), -grad.y * sin(val.y));
  }
} // CosGrad

template <typename T>
__global__ void _SinGrad(const int N, const T* dy, const T* x, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * cos(x[i]);
  }
}

template <>
__global__ void
_SinGrad<half>(const int N, const half* dy, const half* x, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __float2half(__half2float(dy[i]) * cos(__half2float(x[i])));
  }
} // SinGrad

template <>
__global__ void
_SinGrad<half2>(const int N, const half2* dy, const half2* x, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(x[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(grad.x * cos(val.x), grad.y * cos(val.y));
  }
} // SinGrad

template <typename T>
__global__ void _ReciprocalGrad(const int N, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = -dy[i] * math::utils::Square(y[i]);
  }
}

template <>
__global__ void
_ReciprocalGrad<half>(const int N, const half* dy, const half* y, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __float2half(
        -__half2float(dy[i]) * math::utils::Square(__half2float(y[i])));
  }
} // ReciprocalGrad

template <>
__global__ void _ReciprocalGrad<half2>(
    const int N,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] =
        __floats2half2_rn(-grad.x * (val.x * val.x), -grad.y * (val.y * val.y));
  }
} // ReciprocalGrad

template <typename T>
__global__ void _RsqrtGrad(const int N, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = T(-0.5) * dy[i] * math::utils::Cube(y[i]);
  }
}

template <>
__global__ void
_RsqrtGrad<half>(const int N, const half* dy, const half* y, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __float2half(
        -0.5f * __half2float(dy[i]) * math::utils::Cube(__half2float(y[i])));
  }
} // ReciprocalGrad

template <>
__global__ void
_RsqrtGrad<half2>(const int N, const half2* dy, const half2* y, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        -0.5f * grad.x * (val.x * val.x * val.x),
        -0.5f * grad.y * (val.y * val.y * val.y));
  }
} // ReciprocalGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                \
  template <>                                                               \
  void name##Grad<T, CUDAContext>(                                          \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) {      \
    _##name##Grad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, dy, x, dx);                                                      \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(name)                                     \
  template <>                                                                 \
  void name##Grad<float16, CUDAContext>(                                      \
      const int N,                                                            \
      const float16* dy,                                                      \
      const float16* x,                                                       \
      float16* dx,                                                            \
      CUDAContext* ctx) {                                                     \
    if ((N & 1) == 0) {                                                       \
      _##name##Grad<<<                                                        \
          CUDA_BLOCKS(N >> 1),                                                \
          CUDA_THREADS,                                                       \
          0,                                                                  \
          ctx->cuda_stream()>>>(                                              \
          N >> 1,                                                             \
          reinterpret_cast<const half2*>(dy),                                 \
          reinterpret_cast<const half2*>(x),                                  \
          reinterpret_cast<half2*>(dx));                                      \
    } else {                                                                  \
      _##name##Grad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N,                                                                  \
          reinterpret_cast<const half*>(dy),                                  \
          reinterpret_cast<const half*>(x),                                   \
          reinterpret_cast<half*>(dx));                                       \
    }                                                                         \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
