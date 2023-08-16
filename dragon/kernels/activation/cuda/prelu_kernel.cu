#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

/*
 * PRelu Kernels
 */

template <typename T>
__global__ void _PRelu(const int N, const T* x, const T* w, T* y) {
  const T kAlpha = __ldg(w);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > T(0) ? __ldg(x + i) : __ldg(x + i) * kAlpha;
  }
}

template <>
__global__ void
_PRelu<half>(const int N, const half* x, const half* w, half* y) {
#if __CUDA_ARCH__ >= 530
  const half kAlpha = __ldg(w);
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > kZero ? __ldg(x + i) : __ldg(x + i) * kAlpha;
  }
#else
  const float kAlpha = __half2float(__ldg(w));
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(val * kAlpha);
  }
#endif
}

template <>
__global__ void _PRelu<nv_bfloat16>(
    const int N,
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    nv_bfloat16* y) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kAlpha = __ldg(w);
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __ldg(x + i) > kZero ? __ldg(x + i) : __ldg(x + i) * kAlpha;
  }
#else
  const float kAlpha = __bfloat162float(w[0]);
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __bfloat162float(x[i]);
    y[i] = val > 0.f ? x[i] : __float2bfloat16(val * kAlpha);
  }
#endif
}

template <typename T>
__global__ void _PRelu(
    const int NxCxS,
    const int S,
    const int C,
    const StorageOrder order,
    const T* x,
    const T* w,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = order == StorageOrder::NCHW ? i / S % C : i % C;
    y[i] = __ldg(x + i) > T(0) ? __ldg(x + i) : __ldg(x + i) * __ldg(w + j);
  }
}

template <>
__global__ void _PRelu<half>(
    const int NxCxS,
    const int S,
    const int C,
    const StorageOrder order,
    const half* x,
    const half* w,
    half* y) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const auto j = order == StorageOrder::NCHW ? i / S % C : i % C;
    y[i] = __ldg(x + i) > kZero ? __ldg(x + i) : __ldg(x + i) * __ldg(w + j);
  }
#else
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const auto j = order == StorageOrder::NCHW ? i / S % C : i % C;
    const float val = __half2float(__ldg(x + i));
    const float alpha = __half2float(__ldg(w + j));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(val * alpha);
  }
#endif
}

template <>
__global__ void _PRelu<nv_bfloat16>(
    const int NxCxS,
    const int S,
    const int C,
    const StorageOrder order,
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    nv_bfloat16* y) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const auto j = order == StorageOrder::NCHW ? i / S % C : i % C;
    y[i] = __ldg(x + i) > kZero ? __ldg(x + i) : __ldg(x + i) * __ldg(w + j);
  }
#else
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const auto j = order == StorageOrder::NCHW ? i / S % C : i % C;
    const float val = __bfloat162float(x[i]);
    const float alpha = __bfloat162float(w[j]);
    y[i] = val > 0.f ? x[i] : __float2bfloat16(val * alpha);
  }
#endif
}

/*
 * PReluGrad Kernels
 */

template <typename T>
__global__ void
_PReluGrad(const int N, const T* dy, const T* x, const T* w, T* dx) {
  const T kAlpha = __ldg(w);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = x[i] > T(0) ? dy[i] : dy[i] * kAlpha;
  }
}

template <>
__global__ void _PReluGrad<half>(
    const int N,
    const half* dy,
    const half* x,
    const half* w,
    half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kAlpha = __ldg(w);
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = x[i] > kZero ? dy[i] : dy[i] * kAlpha;
  }
#else
  const float kAlpha = __half2float(__ldg(w));
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __half2float(x[i]) > 0.f
        ? dy[i]
        : __float2half(__half2float(dy[i]) * kAlpha);
  }
#endif
}

template <>
__global__ void _PReluGrad<nv_bfloat16>(
    const int N,
    const nv_bfloat16* dy,
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kAlpha = __ldg(w);
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = x[i] > kZero ? dy[i] : dy[i] * kAlpha;
  }
#else
  const float kAlpha = __bfloat162float(w[0]);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = __bfloat162float(x[i]) > 0.f
        ? dy[i]
        : __float2bfloat16(__bfloat162float(dy[i]) * kAlpha);
  }
#endif
}

template <typename T>
__global__ void _PReluGrad(
    const int NxCxS,
    const int S,
    const int C,
    const StorageOrder order,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = order == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = x[i] > T(0) ? dy[i] : dy[i] * __ldg(w + j);
  }
}

template <>
__global__ void _PReluGrad<half>(
    const int NxCxS,
    const int S,
    const int C,
    const StorageOrder order,
    const half* dy,
    const half* x,
    const half* w,
    half* dx) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = order == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = x[i] > kZero ? dy[i] : dy[i] * __ldg(w + j);
  }
#else
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = order == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = __half2float(x[i]) > 0.f
        ? dy[i]
        : __float2half(__half2float(dy[i]) * __half2float(__ldg(w + j)));
  }
#endif
}

template <>
__global__ void _PReluGrad<nv_bfloat16>(
    const int NxCxS,
    const int S,
    const int C,
    const StorageOrder order,
    const nv_bfloat16* dy,
    const nv_bfloat16* x,
    const nv_bfloat16* w,
    nv_bfloat16* dx) {
#if __CUDA_ARCH__ >= 800
  const nv_bfloat16 kZero = __float2bfloat16(0.f);
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = order == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = x[i] > kZero ? dy[i] : dy[i] * __ldg(w + j);
  }
#else
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = order == StorageOrder::NCHW ? i / S % C : i % C;
    dx[i] = __bfloat162float(x[i]) > 0.f
        ? dy[i]
        : __float2bfloat16(__bfloat162float(dy[i]) * __bfloat162float(w[j]));
  }
#endif
}

template <typename T, typename AccT>
__global__ void _PReluWGrad(const int N, const T* dy, const T* x, T* dw) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  const auto less = math::LessFunctor<T>();
  const auto mul = math::MultipliesFunctor<T>();
  const T kZero = convert::To<T>(0.f);
  AccT val = AccT(0);
  CUDA_2D_KERNEL_LOOP2(i, N) {
    val += less(math::utils::LDG(x + i), kZero)
        ? convert::To<AccT>(mul(dy[i], math::utils::LDG(x + i)))
        : AccT(0);
  }
  val = BlockReduce<AccT>(storage).Sum(val);
  if (threadIdx.x == 0) dw[0] = val;
}

template <typename T, typename AccT>
__global__ void _PReluWGrad(
    const int NxS,
    const int S,
    const int C,
    const StorageOrder order,
    const T* dy,
    const T* x,
    T* dw) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  const auto less = math::LessFunctor<T>();
  const auto mul = math::MultipliesFunctor<T>();
  const T kZero = convert::To<T>(0.f);
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, NxS) {
      const int index = (order == StorageOrder::NCHW)
          ? (j / S * C + i) * S + j % S
          : j * C + i;
      val += less(math::utils::LDG(x + index), kZero)
          ? convert::To<AccT>(mul(dy[index], math::utils::LDG(x + index)))
          : AccT(0);
    }
    val = BlockReduce<AccT>(storage).Sum(val);
    if (threadIdx.x == 0) dw[i] = val;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void PRelu<T, CUDAContext>(                                              \
      const int N,                                                         \
      const int S,                                                         \
      const int C,                                                         \
      const string& data_format,                                           \
      const T* x,                                                          \
      const T* w,                                                          \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    using ScalarT = math::Traits<T>::scalar_type;                          \
    const auto NxCxS = N * C * S;                                          \
    if (C > 1) {                                                           \
      _PRelu<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxCxS,                                                           \
          S,                                                               \
          C,                                                               \
          data_format == "NCHW" ? StorageOrder::NCHW : StorageOrder::NHWC, \
          reinterpret_cast<const ScalarT*>(x),                             \
          reinterpret_cast<const ScalarT*>(w),                             \
          reinterpret_cast<ScalarT*>(y));                                  \
    } else {                                                               \
      _PRelu<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxCxS, (const ScalarT*)x, (const ScalarT*)w, (ScalarT*)y);       \
    }                                                                      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                         \
  template <>                                                                  \
  void PReluGrad<T, CUDAContext>(                                              \
      const int N,                                                             \
      const int S,                                                             \
      const int C,                                                             \
      const string& data_format,                                               \
      const T* dy,                                                             \
      const T* x,                                                              \
      const T* w,                                                              \
      T* dx,                                                                   \
      CUDAContext* ctx) {                                                      \
    const auto NxCxS = N * C * S;                                              \
    if (C > 1) {                                                               \
      _PReluGrad<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxCxS,                                                               \
          S,                                                                   \
          C,                                                                   \
          data_format == "NCHW" ? StorageOrder::NCHW : StorageOrder::NHWC,     \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),           \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(x),            \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(w),            \
          reinterpret_cast<math::Traits<T>::scalar_type*>(dx));                \
    } else {                                                                   \
      _PReluGrad<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          NxCxS,                                                               \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),           \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(x),            \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(w),            \
          reinterpret_cast<math::Traits<T>::scalar_type*>(dx));                \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  void PReluWGrad<T, CUDAContext>(                                             \
      const int N,                                                             \
      const int S,                                                             \
      const int C,                                                             \
      const string& data_format,                                               \
      const T* dy,                                                             \
      const T* x,                                                              \
      T* dw,                                                                   \
      CUDAContext* ctx) {                                                      \
    const auto NxS = N * S;                                                    \
    const auto NxCxS = NxS * C;                                                \
    if (C > 1) {                                                               \
      _PReluWGrad<                                                             \
          math::Traits<T>::scalar_type,                                        \
          math::Traits<T>::accumulator_type>                                   \
          <<<C, CUDA_THREADS, 0, ctx->cuda_stream()>>>(                        \
              NxS,                                                             \
              S,                                                               \
              C,                                                               \
              data_format == "NCHW" ? StorageOrder::NCHW : StorageOrder::NHWC, \
              reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),       \
              reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
              reinterpret_cast<math::Traits<T>::scalar_type*>(dw));            \
    } else {                                                                   \
      _PReluWGrad<                                                             \
          math::Traits<T>::scalar_type,                                        \
          math::Traits<T>::accumulator_type>                                   \
          <<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(                        \
              NxCxS,                                                           \
              reinterpret_cast<const math::Traits<T>::scalar_type*>(dy),       \
              reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
              reinterpret_cast<math::Traits<T>::scalar_type*>(dw));            \
    }                                                                          \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
