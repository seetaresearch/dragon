#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) convert::To<AccT>(__ldg(x + i))

template <typename T, typename AccT>
__global__ void
_Softmax(const int NxS, const int S, const int C, const T* x, T* y) {
  __shared__ AccT block_max, block_sum;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_x = x + offset;
    auto* offset_y = y + offset;

    AccT val = convert::To<AccT>(__ldg(offset_x));
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val = max(val, LDG(offset_x, j * S));
    }
    val = BlockReduce<AccT>(storage).Reduce(val, cub::Max());
    if (threadIdx.x == 0) block_max = val;
    __syncthreads();

    val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val += exp(LDG(offset_x, j * S) - block_max);
    }
    val = BlockReduce<AccT>(storage).Sum(val);
    if (threadIdx.x == 0) block_sum = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = exp(LDG(offset_x, k) - block_max);
      offset_y[k] = convert::To<T>(val / block_sum);
    }
  }
}

template <typename T, typename AccT>
__global__ void
_LogSoftmax(const int NxS, const int S, const int C, const T* x, T* y) {
  __shared__ AccT block_max, block_sum;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_x = x + offset;
    auto* offset_y = y + offset;

    AccT val = convert::To<AccT>(__ldg(offset_x));
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val = max(val, LDG(offset_x, j * S));
    }
    val = BlockReduce<AccT>(storage).Reduce(val, cub::Max());
    if (threadIdx.x == 0) block_max = val;
    __syncthreads();

    val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val += exp(LDG(offset_x, j * S) - block_max);
    }
    val = BlockReduce<AccT>(storage).Sum(val);
    if (threadIdx.x == 0) block_sum = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = LDG(offset_x, k) - block_max;
      offset_y[k] = convert::To<T>(val - log(block_sum));
    }
  }
}

template <typename T, typename AccT>
__global__ void _SoftmaxGrad(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  __shared__ AccT block_val;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_dy = dy + offset;
    auto* offset_y = y + offset;
    auto* offset_dx = dx + offset;

    AccT val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val += LDG(offset_dy, k) * LDG(offset_y, k);
    }
    val = BlockReduce<AccT>(storage).Sum(val);
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = LDG(offset_dy, k) - block_val;
      offset_dx[k] = convert::To<T>(val * LDG(offset_y, k));
    }
  }
}

template <typename T, typename AccT>
__global__ void _LogSoftmaxGrad(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  __shared__ AccT block_val;
  __shared__ typename BlockReduce<AccT>::TempStorage storage;
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_dy = dy + offset;
    auto* offset_y = y + offset;
    auto* offset_dx = dx + offset;

    AccT val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val += LDG(offset_dy, j * S);
    }
    val = BlockReduce<AccT>(storage).Sum(val);
    if (threadIdx.x == 0) block_val = val;
    __syncthreads();

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = exp(convert::To<AccT>(offset_y[k])) * block_val;
      offset_dx[k] = convert::To<T>(LDG(offset_dy, k) - val);
    }
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const T* x,                                                     \
      T* y,                                                           \
      CUDAContext* ctx) {                                             \
    const auto NxS = N * S;                                           \
    _##name<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<NxS, CUDA_THREADS, 0, ctx->cuda_stream()>>>(               \
            NxS,                                                      \
            S,                                                        \
            C,                                                        \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),    \
            reinterpret_cast<math::ScalarType<T>::type*>(y));         \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const T* dy,                                                    \
      const T* y,                                                     \
      T* dx,                                                          \
      CUDAContext* ctx) {                                             \
    const auto NxS = N * S;                                           \
    _##name<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<NxS, CUDA_THREADS, 0, ctx->cuda_stream()>>>(               \
            NxS,                                                      \
            S,                                                        \
            C,                                                        \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),   \
            reinterpret_cast<const math::ScalarType<T>::type*>(y),    \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));        \
  }

DEFINE_KERNEL_LAUNCHER(Softmax, float16);
DEFINE_KERNEL_LAUNCHER(Softmax, float);
DEFINE_KERNEL_LAUNCHER(Softmax, double);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
