#include "dragon/kernels/activation/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Elu(const int N, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] =
        __ldg(x + i) > T(0) ? __ldg(x + i) : alpha * (exp(__ldg(x + i)) - T(1));
  }
}

template <>
__global__ void
_Elu<half>(const int N, const float alpha, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(__ldg(x + i));
    y[i] = val > 0.f ? __ldg(x + i) : __float2half(alpha * (exp(val) - 1.f));
  }
}

template <>
__global__ void
_Elu<half2>(const int N, const float alpha, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? val.x : alpha * (exp(val.x) - 1.f),
        val.y > 0.f ? val.y : alpha * (exp(val.y) - 1.f));
  }
}

template <typename T>
__global__ void
_EluGrad(const int N, const float alpha, const T* dy, const T* y, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (__ldg(y + i) > T(0) ? T(1) : alpha + __ldg(y + i));
  }
}

template <>
__global__ void _EluGrad<half>(
    const int N,
    const float alpha,
    const half* dy,
    const half* y,
    half* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(y[i]);
    dx[i] =
        __float2half(__half2float(dy[i]) * (val > 0.f ? 1.f : (alpha + val)));
  }
} // EluGrad

template <>
__global__ void _EluGrad<half2>(
    const int N,
    const float alpha,
    const half2* dy,
    const half2* y,
    half2* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 val = __half22float2(y[i]);
    const float2 grad = __half22float2(dy[i]);
    dx[i] = __floats2half2_rn(
        grad.x * (val.x > 0.f ? 1.f : (alpha + val.x)),
        grad.y * (val.y > 0.f ? 1.f : (alpha + val.y)));
  }
} // EluGrad

} // namespace

template <>
void Elu<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _Elu<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        alpha,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Elu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, alpha, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
}

template <>
void EluGrad<float16, CUDAContext>(
    const int N,
    const float alpha,
    const float16* dy,
    const float16* y,
    float16* dx,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _EluGrad<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        alpha,
        reinterpret_cast<const half2*>(dy),
        reinterpret_cast<const half2*>(y),
        reinterpret_cast<half2*>(dx));
  } else {
    _EluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        alpha,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(y),
        reinterpret_cast<half*>(dx));
  }
} // EluGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void Elu<T, CUDAContext>(                                                 \
      const int N, const float alpha, const T* x, T* y, CUDAContext* ctx) { \
    _Elu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(          \
        N, alpha, x, y);                                                    \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                 \
  template <>                                                          \
  void EluGrad<T, CUDAContext>(                                        \
      const int N,                                                     \
      const float alpha,                                               \
      const T* dy,                                                     \
      const T* y,                                                      \
      T* dx,                                                           \
      CUDAContext* ctx) {                                              \
    _EluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, alpha, dy, y, dx);                                          \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
