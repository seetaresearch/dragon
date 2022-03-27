#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define WARP_ITEMS 16
#define BLOCK_THREADS 1024
#define LDG(x, i) convert::To<AccT>(__ldg(x + i))

template <typename T, typename AccT>
__global__ void _SoftmaxViaWarpReduce(
    const int NxS,
    const int S,
    const int C,
    const T* x,
    T* y) {
  const int warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_warps = gridDim.x * blockDim.y;
  for (int i = warp_idx; i < NxS; i += num_warps) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_x = x + offset;
    auto* offset_y = y + offset;

    AccT val = AccT(-FLT_MAX);
    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      val = max(val, LDG(offset_x, j * S));
    }
    const AccT warp_max = WarpAllReduce<AccT, cub::Max>(val);

    val = AccT(0);
    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      val += exp(LDG(offset_x, j * S) - warp_max);
    }
    const AccT warp_sum = WarpAllReduce<AccT, cub::Sum>(val);

    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      const int k = j * S;
      val = exp(LDG(offset_x, k) - warp_max);
      offset_y[k] = convert::To<T>(val / warp_sum);
    }
  }
}

template <typename T, typename AccT>
__global__ void _LogSoftmaxViaWarpReduce(
    const int NxS,
    const int S,
    const int C,
    const T* x,
    T* y) {
  const int warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_warps = gridDim.x * blockDim.y;
  for (int i = warp_idx; i < NxS; i += num_warps) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_x = x + offset;
    auto* offset_y = y + offset;

    AccT val = AccT(-FLT_MAX);
    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      val = max(val, LDG(offset_x, j * S));
    }
    const AccT warp_max = WarpAllReduce<AccT, cub::Max>(val);

    val = AccT(0);
    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      val += exp(LDG(offset_x, j * S) - warp_max);
    }
    const AccT warp_sum = WarpAllReduce<AccT, cub::Sum>(val);

    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      const int k = j * S;
      val = LDG(offset_x, k) - warp_max;
      offset_y[k] = convert::To<T>(val - log(warp_sum));
    }
  }
}

template <typename T, typename AccT>
__global__ void _SoftmaxViaBlockReduce(
    const int NxS,
    const int S,
    const int C,
    const T* x,
    T* y) {
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_x = x + offset;
    auto* offset_y = y + offset;

    AccT val = AccT(-FLT_MAX);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val = max(val, LDG(offset_x, j * S));
    }
    const AccT block_max = BlockAllReduce<AccT, cub::Max, BLOCK_THREADS>(val);

    val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val += exp(LDG(offset_x, j * S) - block_max);
    }
    const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_THREADS>(val);

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = exp(LDG(offset_x, k) - block_max);
      offset_y[k] = convert::To<T>(val / block_sum);
    }
  }
}

template <typename T, typename AccT>
__global__ void _LogSoftmaxViaBlockReduce(
    const int NxS,
    const int S,
    const int C,
    const T* x,
    T* y) {
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_x = x + offset;
    auto* offset_y = y + offset;

    AccT val = AccT(-FLT_MAX);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val = max(val, LDG(offset_x, j * S));
    }
    const AccT block_max = BlockAllReduce<AccT, cub::Max, BLOCK_THREADS>(val);

    val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val += exp(LDG(offset_x, j * S) - block_max);
    }
    const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_THREADS>(val);

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = LDG(offset_x, k) - block_max;
      offset_y[k] = convert::To<T>(val - log(block_sum));
    }
  }
}

template <typename T, typename AccT>
__global__ void _SoftmaxGradViaWarpReduce(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  const int warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_warps = gridDim.x * blockDim.y;
  for (int i = warp_idx; i < NxS; i += num_warps) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_dy = dy + offset;
    auto* offset_y = y + offset;
    auto* offset_dx = dx + offset;

    AccT val = AccT(0);
    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      const int k = j * S;
      val += LDG(offset_dy, k) * LDG(offset_y, k);
    }
    const AccT warp_sum = WarpAllReduce<AccT, cub::Sum>(val);

    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      const int k = j * S;
      val = LDG(offset_dy, k) - warp_sum;
      offset_dx[k] = convert::To<T>(val * LDG(offset_y, k));
    }
  }
}

template <typename T, typename AccT>
__global__ void _LogSoftmaxGradViaWarpReduce(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  const int warp_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_warps = gridDim.x * blockDim.y;
  for (int i = warp_idx; i < NxS; i += num_warps) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_dy = dy + offset;
    auto* offset_y = y + offset;
    auto* offset_dx = dx + offset;

    AccT val = AccT(0);
    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      val += LDG(offset_dy, j * S);
    }
    const AccT warp_sum = WarpAllReduce<AccT, cub::Sum>(val);

    for (int j = threadIdx.x; j < C; j += CUDA_WARP_SIZE) {
      const int k = j * S;
      val = exp(convert::To<AccT>(offset_y[k])) * warp_sum;
      offset_dx[k] = convert::To<T>(LDG(offset_dy, k) - val);
    }
  }
}

template <typename T, typename AccT>
__global__ void _SoftmaxGradViaBlockReduce(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
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
    const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_THREADS>(val);

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = LDG(offset_dy, k) - block_sum;
      offset_dx[k] = convert::To<T>(val * LDG(offset_y, k));
    }
  }
}

template <typename T, typename AccT>
__global__ void _LogSoftmaxGradViaBlockReduce(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  CUDA_2D_KERNEL_LOOP1(i, NxS) {
    const int offset = (i / S) * C * S + (i % S);
    auto* offset_dy = dy + offset;
    auto* offset_y = y + offset;
    auto* offset_dx = dx + offset;

    AccT val = AccT(0);
    CUDA_2D_KERNEL_LOOP2(j, C) {
      val += LDG(offset_dy, j * S);
    }
    const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_THREADS>(val);

    CUDA_2D_KERNEL_LOOP2(j, C) {
      const int k = j * S;
      val = exp(convert::To<AccT>(offset_y[k])) * block_sum;
      offset_dx[k] = convert::To<T>(LDG(offset_dy, k) - val);
    }
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)                              \
  template <>                                                        \
  void name<T, CUDAContext>(                                         \
      const int N,                                                   \
      const int S,                                                   \
      const int C,                                                   \
      const T* x,                                                    \
      T* y,                                                          \
      CUDAContext* ctx) {                                            \
    const auto NxS = N * S;                                          \
    if (C <= 1024) {                                                 \
      const auto nblocks = math::utils::DivUp<int>(NxS, WARP_ITEMS); \
      _##name##ViaWarpReduce<                                        \
          math::ScalarType<T>::type,                                 \
          math::AccumulatorType<T>::type>                            \
          <<<nblocks,                                                \
             dim3(CUDA_WARP_SIZE, WARP_ITEMS),                       \
             0,                                                      \
             ctx->cuda_stream()>>>(                                  \
              NxS,                                                   \
              S,                                                     \
              C,                                                     \
              reinterpret_cast<const math::ScalarType<T>::type*>(x), \
              reinterpret_cast<math::ScalarType<T>::type*>(y));      \
      return;                                                        \
    }                                                                \
    _##name##ViaBlockReduce<                                         \
        math::ScalarType<T>::type,                                   \
        math::AccumulatorType<T>::type>                              \
        <<<NxS, BLOCK_THREADS, 0, ctx->cuda_stream()>>>(             \
            NxS,                                                     \
            S,                                                       \
            C,                                                       \
            reinterpret_cast<const math::ScalarType<T>::type*>(x),   \
            reinterpret_cast<math::ScalarType<T>::type*>(y));        \
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
    if (C <= 1024) {                                                  \
      const auto nblocks = math::utils::DivUp<int>(NxS, WARP_ITEMS);  \
      _##name##ViaWarpReduce<                                         \
          math::ScalarType<T>::type,                                  \
          math::AccumulatorType<T>::type>                             \
          <<<nblocks,                                                 \
             dim3(CUDA_WARP_SIZE, WARP_ITEMS),                        \
             0,                                                       \
             ctx->cuda_stream()>>>(                                   \
              NxS,                                                    \
              S,                                                      \
              C,                                                      \
              reinterpret_cast<const math::ScalarType<T>::type*>(dy), \
              reinterpret_cast<const math::ScalarType<T>::type*>(y),  \
              reinterpret_cast<math::ScalarType<T>::type*>(dx));      \
      return;                                                         \
    }                                                                 \
    _##name##ViaBlockReduce<                                          \
        math::ScalarType<T>::type,                                    \
        math::AccumulatorType<T>::type>                               \
        <<<NxS, BLOCK_THREADS, 0, ctx->cuda_stream()>>>(              \
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
#undef WARP_ITEMS
#undef BLOCK_THREADS

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
