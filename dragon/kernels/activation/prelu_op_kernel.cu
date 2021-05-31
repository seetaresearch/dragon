#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _PRelu(const int N, const T* x, const T* w, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const AccT val = convert::To<AccT>(__ldg(x + i));
    y[i] = val > AccT(0) ? __ldg(x + i)
                         : convert::To<T>(val * convert::To<AccT>(__ldg(w)));
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _PRelu(
    const int NxCxS,
    const int S,
    const int C,
    const T* x,
    const T* w,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = (kOrder == StorageOrder::NCHW ? (i / S) % C : i % C);
    const AccT val = convert::To<AccT>(__ldg(x + i));
    y[i] = val > AccT(0)
        ? __ldg(x + i)
        : convert::To<T>(val * convert::To<AccT>(__ldg(w + j)));
  }
}

template <typename T, typename AccT>
__global__ void
_PReluGrad(const int N, const T* dy, const T* x, const T* w, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) *
        (convert::To<AccT>(x[i]) > AccT(0) ? AccT(1)
                                           : convert::To<AccT>(__ldg(w))));
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _PReluGrad(
    const int NxCxS,
    const int S,
    const int C,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const int j = (kOrder == StorageOrder::NCHW ? (i / S) % C : i % C);
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) *
        (convert::To<AccT>(x[i]) > AccT(0) ? AccT(1)
                                           : convert::To<AccT>(__ldg(w + j))));
  }
}

template <typename T, typename AccT>
__global__ void _PReluWGrad(const int N, const T* dy, const T* x, T* dw) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  AccT val = AccT(0);
  CUDA_2D_KERNEL_LOOP2(i, N) {
    val += convert::To<AccT>(__ldg(x + i)) < AccT(0)
        ? convert::To<AccT>(dy[i]) * convert::To<AccT>(__ldg(x + i))
        : AccT(0);
  }
  val = BlockReduce<AccT>(storage).Sum(val);
  if (threadIdx.x == 0) {
    dw[0] = convert::To<T>(val);
  }
}

template <typename T, typename AccT, StorageOrder kOrder>
__global__ void _PReluWGrad(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* x,
    T* dw) {
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, C) {
    AccT val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, NxS) {
      const int index =
          (kOrder == StorageOrder::NCHW ? (j / S * C + i) * S + j % S
                                        : j * C + i);
      val += convert::To<AccT>(__ldg(x + index)) < AccT(0)
          ? convert::To<AccT>(dy[index]) * convert::To<AccT>(__ldg(x + index))
          : AccT(0);
    }
    val = BlockReduce<AccT>(storage).Sum(val);
    if (threadIdx.x == 0) {
      dw[i] = convert::To<T>(val);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_CWISE_PRELU_KERNEL(name, T, AccT, kBlocks, kThreads, ...) \
  if (data_format == "NCHW") {                                             \
    name<T, AccT, StorageOrder::NCHW>                                      \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);       \
  } else if (data_format == "NHWC") {                                      \
    name<T, AccT, StorageOrder::NHWC>                                      \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);       \
  } else {                                                                 \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                   \
  }

#define DEFINE_KERNEL_LAUNCHER(T)                                        \
  template <>                                                            \
  void PRelu<T, CUDAContext>(                                            \
      const int N,                                                       \
      const int S,                                                       \
      const int C,                                                       \
      const string& data_format,                                         \
      const T* x,                                                        \
      const T* w,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    const auto NxCxS = N * C * S;                                        \
    if (C > 1) {                                                         \
      DISPATCH_CWISE_PRELU_KERNEL(                                       \
          _PRelu,                                                        \
          math::ScalarType<T>::type,                                     \
          math::AccmulatorType<T>::type,                                 \
          CUDA_BLOCKS(NxCxS),                                            \
          CUDA_THREADS,                                                  \
          NxCxS,                                                         \
          S,                                                             \
          C,                                                             \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),         \
          reinterpret_cast<const math::ScalarType<T>::type*>(w),         \
          reinterpret_cast<math::ScalarType<T>::type*>(y));              \
    } else {                                                             \
      _PRelu<math::ScalarType<T>::type, math::AccmulatorType<T>::type>   \
          <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
              NxCxS,                                                     \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),     \
              reinterpret_cast<const math::ScalarType<T>::type*>(w),     \
              reinterpret_cast<math::ScalarType<T>::type*>(y));          \
    }                                                                    \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                      \
  template <>                                                               \
  void PReluGrad<T, CUDAContext>(                                           \
      const int N,                                                          \
      const int S,                                                          \
      const int C,                                                          \
      const string& data_format,                                            \
      const T* dy,                                                          \
      const T* x,                                                           \
      const T* w,                                                           \
      T* dx,                                                                \
      CUDAContext* ctx) {                                                   \
    const auto NxCxS = N * C * S;                                           \
    if (C > 1) {                                                            \
      DISPATCH_CWISE_PRELU_KERNEL(                                          \
          _PReluGrad,                                                       \
          math::ScalarType<T>::type,                                        \
          math::AccmulatorType<T>::type,                                    \
          CUDA_BLOCKS(NxCxS),                                               \
          CUDA_THREADS,                                                     \
          NxCxS,                                                            \
          S,                                                                \
          C,                                                                \
          reinterpret_cast<const math::ScalarType<T>::type*>(dy),           \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),            \
          reinterpret_cast<const math::ScalarType<T>::type*>(w),            \
          reinterpret_cast<math::ScalarType<T>::type*>(dx));                \
    } else {                                                                \
      _PReluGrad<math::ScalarType<T>::type, math::AccmulatorType<T>::type>  \
          <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
              NxCxS,                                                        \
              reinterpret_cast<const math::ScalarType<T>::type*>(dy),       \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),        \
              reinterpret_cast<const math::ScalarType<T>::type*>(w),        \
              reinterpret_cast<math::ScalarType<T>::type*>(dx));            \
    }                                                                       \
  }                                                                         \
  template <>                                                               \
  void PReluWGrad<T, CUDAContext>(                                          \
      const int N,                                                          \
      const int S,                                                          \
      const int C,                                                          \
      const string& data_format,                                            \
      const T* dy,                                                          \
      const T* x,                                                           \
      T* dw,                                                                \
      CUDAContext* ctx) {                                                   \
    const auto NxS = N * S;                                                 \
    const auto NxCxS = NxS * C;                                             \
    if (C > 1) {                                                            \
      DISPATCH_CWISE_PRELU_KERNEL(                                          \
          _PReluWGrad,                                                      \
          math::ScalarType<T>::type,                                        \
          math::AccmulatorType<T>::type,                                    \
          C,                                                                \
          CUDA_THREADS,                                                     \
          NxS,                                                              \
          S,                                                                \
          C,                                                                \
          reinterpret_cast<const math::ScalarType<T>::type*>(dy),           \
          reinterpret_cast<const math::ScalarType<T>::type*>(x),            \
          reinterpret_cast<math::ScalarType<T>::type*>(dw));                \
    } else {                                                                \
      _PReluWGrad<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
          <<<1, CUDA_THREADS, 0, ctx->cuda_stream()>>>(                     \
              NxCxS,                                                        \
              reinterpret_cast<const math::ScalarType<T>::type*>(dy),       \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),        \
              reinterpret_cast<math::ScalarType<T>::type*>(dw));            \
    }                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_CWISE_PRELU_KERNEL

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
