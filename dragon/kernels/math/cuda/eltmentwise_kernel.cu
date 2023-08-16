#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

/*
 * CosGrad Kernels
 */

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
#if __CUDA_ARCH__ >= 530
    dx[i] = -dy[i] * hsin(x[i]);
#else
    dx[i] = __float2half(-__half2float(dy[i]) * sinf(__half2float(x[i])));
#endif
  }
}

template <>
__global__ void
_CosGrad<half2>(const int N, const half2* dy, const half2* x, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dx[i] = -dy[i] * h2sin(x[i]);
#else
    const float2 val = __half22float2(x[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(-grad.x * sinf(val.x), -grad.y * sinf(val.y));
#endif
  }
}

template <>
__global__ void _CosGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* x,
    nv_bfloat16* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = -dy[i] * hsin(x[i]);
#else
    dx[i] = __float2bfloat16(
        -__bfloat162float(dy[i]) * sinf(__bfloat162float(x[i])));
#endif
  }
}

template <>
__global__ void _CosGrad<nv_bfloat162>(
    const int N,
    const nv_bfloat162* dy,
    const nv_bfloat162* x,
    nv_bfloat162* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = -dy[i] * h2sin(x[i]);
#else
    const float2 val = convert::To<float2>(x[i]);
    const float2 grad = convert::To<float2>(dy[i]);
    dx[i] = __floats2bfloat162_rn(-grad.x * sin(val.x), -grad.y * sin(val.y));
#endif
  }
}

/*
 * SinGrad Kernels
 */

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
#if __CUDA_ARCH__ >= 530
    dx[i] = dy[i] * hcos(x[i]);
#else
    dx[i] = __float2half(__half2float(dy[i]) * cosf(__half2float(x[i])));
#endif
  }
}

template <>
__global__ void
_SinGrad<half2>(const int N, const half2* dy, const half2* x, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dx[i] = dy[i] * h2cos(x[i]);
#else
    const float2 val = __half22float2(x[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(grad.x * cosf(val.x), grad.y * cosf(val.y));
#endif
  }
}

template <>
__global__ void _SinGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* x,
    nv_bfloat16* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = dy[i] * hcos(x[i]);
#else
    dx[i] = __float2bfloat16(
        __bfloat162float(dy[i]) * cosf(__bfloat162float(x[i])));
#endif
  }
}

template <>
__global__ void _SinGrad<nv_bfloat162>(
    const int N,
    const nv_bfloat162* dy,
    const nv_bfloat162* x,
    nv_bfloat162* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = dy[i] * h2cos(x[i]);
#else
    const float2 val = convert::To<float2>(x[i]);
    const float2 grad = convert::To<float2>(dy[i]);
    dx[i] = __floats2bfloat162_rn(grad.x * cosf(val.x), grad.y * cosf(val.y));
#endif
  }
}

/*
 * ReciprocalGrad Kernels
 */

template <typename T>
__global__ void _ReciprocalGrad(const int N, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = -dy[i] * math::utils::Sqr(y[i]);
  }
}

template <>
__global__ void
_ReciprocalGrad<half>(const int N, const half* dy, const half* y, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dx[i] = -dy[i] * math::utils::Sqr(y[i]);
#else
    dx[i] = __float2half(
        -__half2float(dy[i]) * math::utils::Sqr(__half2float(y[i])));
#endif
  }
}

template <>
__global__ void _ReciprocalGrad<half2>(
    const int N,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dx[i] = -dy[i] * math::utils::Sqr(y[i]);
#else
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(-grad.x * val.x * val.x, -grad.y * val.y * val.y);
#endif
  }
}

template <>
__global__ void _ReciprocalGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = -dy[i] * math::utils::Sqr(y[i]);
#else
    dx[i] = __float2bfloat16(
        -__bfloat162float(dy[i]) * math::utils::Sqr(__bfloat162float(y[i])));
#endif
  }
}

template <>
__global__ void _ReciprocalGrad<nv_bfloat162>(
    const int N,
    const nv_bfloat162* dy,
    const nv_bfloat162* y,
    nv_bfloat162* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = -dy[i] * math::utils::Sqr(y[i]);
#else
    const float2 val = convert::To<float2>(y[i]);
    const float2 grad = convert::To<float2>(dy[i]);
    dx[i] = __floats2bfloat162_rn(
        -grad.x * (val.x * val.x), -grad.y * (val.y * val.y));
#endif
  }
}

/*
 * RsqrtGrad Kernels
 */

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
#if __CUDA_ARCH__ >= 530
    dx[i] = __float2half(-0.5f) * dy[i] * math::utils::Cube(y[i]);
#else
    dx[i] = __float2half(
        -0.5f * __half2float(dy[i]) * math::utils::Cube(__half2float(y[i])));
#endif
  }
}

template <>
__global__ void
_RsqrtGrad<half2>(const int N, const half2* dy, const half2* y, half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __float2half2_rn(-0.5f) * dy[i] * math::utils::Cube(y[i]);
#else
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        -0.5f * grad.x * (val.x * val.x * val.x),
        -0.5f * grad.y * (val.y * val.y * val.y));
#endif
  }
}

template <>
__global__ void _RsqrtGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* y,
    nv_bfloat16* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 800
    dx[i] = __float2bfloat16(-0.5f) * dy[i] * math::utils::Cube(y[i]);
#else
    dx[i] = __float2bfloat16(
        -0.5f * __bfloat162float(dy[i]) *
        math::utils::Cube(__bfloat162float(y[i])));
#endif
  }
}

template <>
__global__ void _RsqrtGrad<nv_bfloat162>(
    const int N,
    const nv_bfloat162* dy,
    const nv_bfloat162* y,
    nv_bfloat162* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __float2bfloat162_rn(-0.5f) * dy[i] * math::utils::Cube(y[i]);
#else
    const float2 val = convert::To<float2>(y[i]);
    const float2 grad = convert::To<float2>(dy[i]);
    dx[i] = __floats2bfloat162_rn(
        -0.5f * grad.x * (val.x * val.x * val.x),
        -0.5f * grad.y * (val.y * val.y * val.y));
#endif
  }
}

} // namespace

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                                 \
  void name##Grad<T, CUDAContext>(                                            \
      const int N, const T* dy, const T* x, T* dx, CUDAContext* ctx) {        \
    if ((N & 1) == 0 && math::Traits<T>::HasPack2()) {                        \
      _##name##Grad<<<                                                        \
          CUDA_BLOCKS(N >> 1),                                                \
          CUDA_THREADS,                                                       \
          0,                                                                  \
          ctx->cuda_stream()>>>(                                              \
          N >> 1,                                                             \
          reinterpret_cast<const math::Traits<T>::scalar2_type*>(dy),         \
          reinterpret_cast<const math::Traits<T>::scalar2_type*>(x),          \
          reinterpret_cast<math::Traits<T>::scalar2_type*>(dx));              \
    } else {                                                                  \
      _##name##Grad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N,                                                                  \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),          \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(x),           \
          reinterpret_cast<math::Traits<T>::scalar_type*>(dx));               \
    }                                                                         \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
