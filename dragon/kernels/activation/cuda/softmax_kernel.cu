#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

#define WARP_ITEMS 16
#define WARP_SIZE 32
#define BLOCK_SIZE 1024
#define LDG(x, i) convert::To<AccT>(__ldg(x + i))

template <typename T, typename AccT>
__global__ void
_WarpSoftmax(const int NxS, const int S, const int C, const T* x, T* y) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= NxS) return;
  const int offset = (i / S) * C * S + (i % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    val = max(val, LDG(offset_x, j * S));
  }
  const AccT warp_max = WarpAllReduce<AccT, cub::Max, WARP_SIZE>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    val += exp(LDG(offset_x, j * S) - warp_max);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, WARP_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    const int k = j * S;
    val = exp(LDG(offset_x, k) - warp_max);
    offset_y[k] = convert::To<T>(val / warp_sum);
  }
}

template <typename T, typename AccT>
__global__ void
_WarpLogSoftmax(const int NxS, const int S, const int C, const T* x, T* y) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= NxS) return;
  const int offset = (i / S) * C * S + (i % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    val = max(val, LDG(offset_x, j * S));
  }
  const AccT warp_max = WarpAllReduce<AccT, cub::Max, WARP_SIZE>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    val += exp(LDG(offset_x, j * S) - warp_max);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, WARP_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    const int k = j * S;
    val = LDG(offset_x, k) - warp_max;
    offset_y[k] = convert::To<T>(val - log(warp_sum));
  }
}

template <typename T, typename AccT>
__global__ void _BlockSoftmax(const int S, const int C, const T* x, T* y) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    val = max(val, LDG(offset_x, j * S));
  }
  const AccT block_max = BlockAllReduce<AccT, cub::Max, BLOCK_SIZE>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    val += exp(LDG(offset_x, j * S) - block_max);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    const int k = j * S;
    val = exp(LDG(offset_x, k) - block_max);
    offset_y[k] = convert::To<T>(val / block_sum);
  }
}

template <typename T, typename AccT>
__global__ void _BlockLogSoftmax(const int S, const int C, const T* x, T* y) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    val = max(val, LDG(offset_x, j * S));
  }
  const AccT block_max = BlockAllReduce<AccT, cub::Max, BLOCK_SIZE>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    val += exp(LDG(offset_x, j * S) - block_max);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    const int k = j * S;
    val = LDG(offset_x, k) - block_max;
    offset_y[k] = convert::To<T>(val - log(block_sum));
  }
}

template <typename T, typename AccT>
__global__ void _WarpSoftmaxGrad(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= NxS) return;
  const int offset = (i / S) * C * S + (i % S);
  auto *offset_dy = dy + offset, *offset_y = y + offset;
  auto* offset_dx = dx + offset;
  AccT val = AccT(0);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    const int k = j * S;
    val += LDG(offset_dy, k) * LDG(offset_y, k);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, WARP_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    const int k = j * S;
    val = LDG(offset_dy, k) - warp_sum;
    offset_dx[k] = convert::To<T>(val * LDG(offset_y, k));
  }
}

template <typename T, typename AccT>
__global__ void _WarpLogSoftmaxGrad(
    const int NxS,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= NxS) return;
  const int offset = (i / S) * C * S + (i % S);
  auto *offset_dy = dy + offset, *offset_y = y + offset;
  auto* offset_dx = dx + offset;
  AccT val = AccT(0);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    val += LDG(offset_dy, j * S);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, WARP_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += WARP_SIZE) {
    const int k = j * S;
    val = exp(convert::To<AccT>(offset_y[k])) * warp_sum;
    offset_dx[k] = convert::To<T>(LDG(offset_dy, k) - val);
  }
}

template <typename T, typename AccT>
__global__ void
_BlockSoftmaxGrad(const int S, const int C, const T* dy, const T* y, T* dx) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto *offset_dy = dy + offset, *offset_y = y + offset;
  auto* offset_dx = dx + offset;
  AccT val = AccT(0);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    const int k = j * S;
    val += LDG(offset_dy, k) * LDG(offset_y, k);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    const int k = j * S;
    val = LDG(offset_dy, k) - block_sum;
    offset_dx[k] = convert::To<T>(val * LDG(offset_y, k));
  }
}

template <typename T, typename AccT>
__global__ void
_BlockLogSoftmaxGrad(const int S, const int C, const T* dy, const T* y, T* dx) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto *offset_dy = dy + offset, *offset_y = y + offset;
  auto* offset_dx = dx + offset;
  AccT val = AccT(0);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    val += LDG(offset_dy, j * S);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, BLOCK_SIZE>(val);
  for (int j = threadIdx.x; j < C; j += BLOCK_SIZE) {
    const int k = j * S;
    val = exp(convert::To<AccT>(offset_y[k])) * block_sum;
    offset_dx[k] = convert::To<T>(LDG(offset_dy, k) - val);
  }
}

#undef LDG

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                       \
  template <>                                                                 \
  void name<T, CUDAContext>(                                                  \
      const int N,                                                            \
      const int S,                                                            \
      const int C,                                                            \
      const T* x,                                                             \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    const auto NxS = N * S;                                                   \
    if (C <= 1024) {                                                          \
      const auto block_threads = dim3(WARP_SIZE, std::min(NxS, WARP_ITEMS));  \
      const auto num_blocks = math::utils::DivUp<int>(NxS, block_threads.y);  \
      _Warp##name<math::ScalarType<T>::type, math::AccumulatorType<T>::type>  \
          <<<num_blocks, block_threads, 0, ctx->cuda_stream()>>>(             \
              NxS,                                                            \
              S,                                                              \
              C,                                                              \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),          \
              reinterpret_cast<math::ScalarType<T>::type*>(y));               \
    } else {                                                                  \
      _Block##name<math::ScalarType<T>::type, math::AccumulatorType<T>::type> \
          <<<NxS, BLOCK_SIZE, 0, ctx->cuda_stream()>>>(                       \
              S,                                                              \
              C,                                                              \
              reinterpret_cast<const math::ScalarType<T>::type*>(x),          \
              reinterpret_cast<math::ScalarType<T>::type*>(y));               \
    }                                                                         \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                  \
  template <>                                                                 \
  void name<T, CUDAContext>(                                                  \
      const int N,                                                            \
      const int S,                                                            \
      const int C,                                                            \
      const T* dy,                                                            \
      const T* y,                                                             \
      T* dx,                                                                  \
      CUDAContext* ctx) {                                                     \
    const auto NxS = N * S;                                                   \
    if (C <= 1024) {                                                          \
      const auto block_threads = dim3(WARP_SIZE, std::min(NxS, WARP_ITEMS));  \
      const auto num_blocks = math::utils::DivUp<int>(NxS, block_threads.y);  \
      _Warp##name<math::ScalarType<T>::type, math::AccumulatorType<T>::type>  \
          <<<num_blocks, block_threads, 0, ctx->cuda_stream()>>>(             \
              NxS,                                                            \
              S,                                                              \
              C,                                                              \
              reinterpret_cast<const math::ScalarType<T>::type*>(dy),         \
              reinterpret_cast<const math::ScalarType<T>::type*>(y),          \
              reinterpret_cast<math::ScalarType<T>::type*>(dx));              \
    } else {                                                                  \
      _Block##name<math::ScalarType<T>::type, math::AccumulatorType<T>::type> \
          <<<NxS, BLOCK_SIZE, 0, ctx->cuda_stream()>>>(                       \
              S,                                                              \
              C,                                                              \
              reinterpret_cast<const math::ScalarType<T>::type*>(dy),         \
              reinterpret_cast<const math::ScalarType<T>::type*>(y),          \
              reinterpret_cast<math::ScalarType<T>::type*>(dx));              \
    }                                                                         \
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
#undef WARP_SIZE
#undef BLOCK_SIZE

} // namespace kernels

} // namespace dragon
