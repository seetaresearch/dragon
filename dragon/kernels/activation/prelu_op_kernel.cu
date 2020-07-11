#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cub_device.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _PRelu(const int nthreads, const T* x, const T* w, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) > T(0) ? __ldg(x + i) : __ldg(x + i) * __ldg(w);
#else
    y[i] = x[i] > T(0) ? x[i] : x[i] * w[0];
#endif
  }
}

template <>
__global__ void
_PRelu<half>(const int nthreads, const half* x, const half* w, half* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hgt(__ldg(x + i), kZero) ? __ldg(x + i)
                                      : __hmul(__ldg(x + i), __ldg(w));
#endif
  }
}

template <typename T>
__global__ void _PReluNCHW(
    const int nthreads,
    const int C,
    const int S,
    const T* x,
    const T* w,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] = __ldg(x + i) > T(0) ? __ldg(x + i)
                               : __ldg(x + i) * __ldg(w + ((i / S) % C));
#else
    y[i] = x[i] > T(0) ? x[i] : x[i] * w[(i / S) % C];
#endif
  }
}

template <>
__global__ void _PReluNCHW<half>(
    const int nthreads,
    const int C,
    const int S,
    const half* x,
    const half* w,
    half* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hgt(__ldg(x + i), kZero)
        ? __ldg(x + i)
        : __hmul(__ldg(x + i), __ldg(w + ((i / S) % C)));
#endif
  }
}

template <typename T>
__global__ void
_PReluNHWC(const int nthreads, const int C, const T* x, const T* w, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    y[i] =
        __ldg(x + i) > T(0) ? __ldg(x + i) : __ldg(x + i) * __ldg(w + (i % C));
#else
    y[i] = x[i] > T(0) ? x[i] : x[i] * w[i % C];
#endif
  }
}

template <>
__global__ void _PReluNHWC<half>(
    const int nthreads,
    const int C,
    const half* x,
    const half* w,
    half* y) {
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    y[i] = __hgt(__ldg(x + i), kZero)
        ? __ldg(x + i)
        : __hmul(__ldg(x + i), __ldg(w + (i % C)));
#endif
  }
}

template <typename T>
__global__ void
_PReluGrad(const int nthreads, const T* dy, const T* x, const T* w, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = dy[i] * (__ldg(x + i) > T(0) ? T(1) : __ldg(w));
#else
    dx[i] = dy[i] * (x[i] > T(0) ? T(1) : w[0]);
#endif
  }
}

template <>
__global__ void _PReluGrad<half>(
    const int nthreads,
    const half* dy,
    const half* x,
    const half* w,
    half* dx) {
  const half kOne = __float2half(1.f);
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = __hmul(dy[i], (__hgt(__ldg(x + i), kZero) ? kOne : __ldg(w)));
#endif
  }
}

template <typename T>
__global__ void _PReluGradNCHW(
    const int nthreads,
    const int C,
    const int S,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = dy[i] * (__ldg(x + i) > T(0) ? T(1) : __ldg(w + ((i / S) % C)));
#else
    dx[i] = dy[i] * (x[i] > T(0) ? T(1) : w[(i / S) % C]);
#endif
  }
}

template <>
__global__ void _PReluGradNCHW<half>(
    const int nthreads,
    const int C,
    const int S,
    const half* dy,
    const half* x,
    const half* w,
    half* dx) {
  const half kOne = __float2half(1.f);
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] =
        dy[i] * (__hgt(__ldg(x + i), kZero) ? kOne : __ldg(w + ((i / S) % C)));
#endif
  }
}

template <typename T>
__global__ void _PReluGradNHWC(
    const int nthreads,
    const int C,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
    dx[i] = dy[i] * (__ldg(x + i) > T(0) ? T(1) : __ldg(w + (i % C)));
#else
    dx[i] = dy[i] * (x[i] > T(0) ? T(1) : w[i % C]);
#endif
  }
}

template <>
__global__ void _PReluGradNHWC<half>(
    const int nthreads,
    const int C,
    const half* dy,
    const half* x,
    const half* w,
    half* dx) {
  const half kOne = __float2half(1.f);
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
    dx[i] = dy[i] * (__hgt(__ldg(x + i), kZero) ? kOne : __ldg(w + (i % C)));
#endif
  }
}

template <typename T>
__global__ void _PReluWGrad(const int N, const T* dy, const T* x, T* dw) {
  __shared__ typename BlockReduce<T>::TempStorage storage;

  T val = T(0);
  CUDA_2D_KERNEL_LOOP2(i, N) {
    val += x[i] < T(0) ? dy[i] * x[i] : T(0);
  }

  val = BlockReduce<T>(storage).Sum(val);
  if (threadIdx.x == 0) *dw = val;
}

template <>
__global__ void
_PReluWGrad<half>(const int N, const half* dy, const half* x, half* dw) {
  const half kZero = __float2half(0.f);
  __shared__ typename BlockReduce<float>::TempStorage storage;

  float val = 0.f;
  CUDA_2D_KERNEL_LOOP2(i, N) {
#if __CUDA_ARCH__ >= 530
    val += __hlt(x[i], kZero) ? __half2float(__hmul(dy[i], x[i])) : 0.f;
#endif
  }

  val = BlockReduce<float>(storage).Sum(val);
  if (threadIdx.x == 0) *dw = __float2half(val);
}

template <typename T>
__global__ void _PReluWGradNCHW(
    const int NS,
    const int C,
    const int S,
    const T* dy,
    const T* x,
    T* dw) {
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    T val = T(0);
    CUDA_2D_KERNEL_LOOP2(j, NS) {
      const int yi = ((j / S) * C + i) * S + j % S;
      val += x[yi] < T(0) ? dy[yi] * x[yi] : T(0);
    }
    val = BlockReduce<T>(storage).Sum(val);
    if (threadIdx.x == 0) dw[i] = val;
  }
}

template <>
__global__ void _PReluWGradNCHW<half>(
    const int NS,
    const int C,
    const int S,
    const half* dy,
    const half* x,
    half* dw) {
  const half kZero = __float2half(0.f);
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    float val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, NS) {
#if __CUDA_ARCH__ >= 530
      const int yi = ((j / S) * C + i) * S + j % S;
      val += __hlt(x[yi], kZero) ? __half2float(__hmul(dy[yi], x[yi])) : 0.f;
#endif
    }
    val = BlockReduce<float>(storage).Sum(val);
    if (threadIdx.x == 0) dw[i] = __float2half(val);
  }
}

template <typename T>
__global__ void
_PReluWGradNHWC(const int NS, const int C, const T* dy, const T* x, T* dw) {
  __shared__ typename BlockReduce<T>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    T val = T(0);
    CUDA_2D_KERNEL_LOOP2(j, NS) {
      const int yi = j * C + i;
      val += x[yi] < 0 ? dy[yi] * x[yi] : T(0);
    }
    val = BlockReduce<T>(storage).Sum(val);
    if (threadIdx.x == 0) dw[i] = val;
  }
}

template <>
__global__ void _PReluWGradNHWC<half>(
    const int NS,
    const int C,
    const half* dy,
    const half* x,
    half* dw) {
  const half kZero = __float2half(0.f);
  __shared__ typename BlockReduce<float>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    float val = 0.f;
    CUDA_2D_KERNEL_LOOP2(j, NS) {
#if __CUDA_ARCH__ >= 530
      const int yi = j * C + i;
      val += __hlt(x[yi], kZero) ? __half2float(__hmul(dy[yi], x[yi])) : 0.f;
#endif
    }
    val = BlockReduce<float>(storage).Sum(val);
    if (threadIdx.x == 0) dw[i] = __float2half(val);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void PRelu<float16, CUDAContext>(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const float16* x,
    const float16* w,
    float16* y,
    CUDAContext* ctx) {
  if (C > 1) {
    if (data_format == "NCHW") {
      _PReluNCHW<<<
          CUDA_BLOCKS(N * C * S),
          CUDA_THREADS,
          0,
          ctx->cuda_stream()>>>(
          N * C * S,
          C,
          S,
          reinterpret_cast<const half*>(x),
          reinterpret_cast<const half*>(w),
          reinterpret_cast<half*>(y));
    } else if (data_format == "NHWC") {
      _PReluNHWC<<<
          CUDA_BLOCKS(N * C * S),
          CUDA_THREADS,
          0,
          ctx->cuda_stream()>>>(
          N * C * S,
          C,
          reinterpret_cast<const half*>(x),
          reinterpret_cast<const half*>(w),
          reinterpret_cast<half*>(y));
    } else {
      LOG(FATAL) << "Unknown data format: " << data_format;
    }
  } else {
    _PRelu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(w),
        reinterpret_cast<half*>(y));
  }
}

template <>
void PReluGrad<float16, CUDAContext>(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const float16* dy,
    const float16* x,
    const float16* w,
    float16* dx,
    CUDAContext* ctx) {
  if (C > 1) {
    if (data_format == "NCHW") {
      _PReluGradNCHW<<<
          CUDA_BLOCKS(N * C * S),
          CUDA_THREADS,
          0,
          ctx->cuda_stream()>>>(
          N * C * S,
          C,
          S,
          reinterpret_cast<const half*>(dy),
          reinterpret_cast<const half*>(x),
          reinterpret_cast<const half*>(w),
          reinterpret_cast<half*>(dx));
    } else if (data_format == "NHWC") {
      _PReluGradNHWC<<<
          CUDA_BLOCKS(N * C * S),
          CUDA_THREADS,
          0,
          ctx->cuda_stream()>>>(
          N * C * S,
          C,
          reinterpret_cast<const half*>(dy),
          reinterpret_cast<const half*>(x),
          reinterpret_cast<const half*>(w),
          reinterpret_cast<half*>(dx));
    } else {
      LOG(FATAL) << "Unknown data format: " << data_format;
    }
  } else {
    _PReluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(w),
        reinterpret_cast<half*>(dx));
  }
} // PReluGrad

template <>
void PReluWGrad<float16, CUDAContext>(
    const int N,
    const int C,
    const int S,
    const string& data_format,
    const float16* dy,
    const float16* x,
    float16* dw,
    CUDAContext* ctx) {
  if (C > 1) {
    if (data_format == "NCHW") {
      _PReluWGradNCHW<<<
          CUDA_2D_BLOCKS(C),
          CUDA_THREADS,
          0,
          ctx->cuda_stream()>>>(
          N * S,
          C,
          S,
          reinterpret_cast<const half*>(dy),
          reinterpret_cast<const half*>(x),
          reinterpret_cast<half*>(dw));
    } else if (data_format == "NHWC") {
      _PReluWGradNHWC<<<
          CUDA_2D_BLOCKS(C),
          CUDA_THREADS,
          0,
          ctx->cuda_stream()>>>(
          N * S,
          C,
          reinterpret_cast<const half*>(dy),
          reinterpret_cast<const half*>(x),
          reinterpret_cast<half*>(dw));
    } else {
      LOG(FATAL) << "Unknown data format: " << data_format;
    }
  } else {
    _PReluWGrad<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(dw));
  }
} // PReluWGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void PRelu<T, CUDAContext>(                                          \
      const int N,                                                     \
      const int C,                                                     \
      const int S,                                                     \
      const string& data_format,                                       \
      const T* x,                                                      \
      const T* w,                                                      \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    if (C > 1) {                                                       \
      if (data_format == "NCHW") {                                     \
        _PReluNCHW<<<                                                  \
            CUDA_BLOCKS(N* C* S),                                      \
            CUDA_THREADS,                                              \
            0,                                                         \
            ctx->cuda_stream()>>>(N * C * S, C, S, x, w, y);           \
      } else if (data_format == "NHWC") {                              \
        _PReluNHWC<<<                                                  \
            CUDA_BLOCKS(N* C* S),                                      \
            CUDA_THREADS,                                              \
            0,                                                         \
            ctx->cuda_stream()>>>(N * C * S, C, x, w, y);              \
      } else {                                                         \
        LOG(FATAL) << "Unknown data format: " << data_format;          \
      }                                                                \
    } else {                                                           \
      _PRelu<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N, x, w, y);                                                 \
    }                                                                  \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void PReluGrad<T, CUDAContext>(                                            \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const T* dy,                                                           \
      const T* x,                                                            \
      const T* w,                                                            \
      T* dx,                                                                 \
      CUDAContext* ctx) {                                                    \
    if (C > 1) {                                                             \
      if (data_format == "NCHW") {                                           \
        _PReluGradNCHW<<<                                                    \
            CUDA_BLOCKS(N* C* S),                                            \
            CUDA_THREADS,                                                    \
            0,                                                               \
            ctx->cuda_stream()>>>(N * C * S, C, S, dy, x, w, dx);            \
      } else if (data_format == "NHWC") {                                    \
        _PReluGradNHWC<<<                                                    \
            CUDA_BLOCKS(N* C* S),                                            \
            CUDA_THREADS,                                                    \
            0,                                                               \
            ctx->cuda_stream()>>>(N * C * S, C, dy, x, w, dx);               \
      } else {                                                               \
        LOG(FATAL) << "Unknown data format: " << data_format;                \
      }                                                                      \
    } else {                                                                 \
      _PReluGrad<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
          N, dy, x, w, dx);                                                  \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void PReluWGrad<T, CUDAContext>(                                           \
      const int N,                                                           \
      const int C,                                                           \
      const int S,                                                           \
      const string& data_format,                                             \
      const T* dy,                                                           \
      const T* x,                                                            \
      T* dw,                                                                 \
      CUDAContext* ctx) {                                                    \
    if (C > 1) {                                                             \
      if (data_format == "NCHW") {                                           \
        _PReluWGradNCHW<<<                                                   \
            CUDA_2D_BLOCKS(C),                                               \
            CUDA_THREADS,                                                    \
            0,                                                               \
            ctx->cuda_stream()>>>(N * S, C, S, dy, x, dw);                   \
      } else if (data_format == "NHWC") {                                    \
        _PReluWGradNHWC<<<                                                   \
            CUDA_2D_BLOCKS(C),                                               \
            CUDA_THREADS,                                                    \
            0,                                                               \
            ctx->cuda_stream()>>>(N * S, C, dy, x, dw);                      \
      } else {                                                               \
        LOG(FATAL) << "Unknown data format: " << data_format;                \
      }                                                                      \
    } else {                                                                 \
      _PReluWGrad<<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(N, dy, x, dw); \
    }                                                                        \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
