#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

constexpr int kBlockSize = 1024;
constexpr int kWarpSize = 32;
constexpr int kWarpItems = 16;

template <typename T, typename AccT>
__global__ void
_WarpSoftmax(const int NxS, const int S, const int C, const T* x, T* y) {
  const int i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i >= NxS) return;
  const int offset = (i / S) * C * S + (i % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    val = max(val, math::utils::LDGC<AccT>(offset_x + j * S));
  }
  const AccT warp_max = WarpAllReduce<AccT, cub::Max, kWarpSize>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    val += exp(math::utils::LDGC<AccT>(offset_x + j * S) - warp_max);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, kWarpSize>(val);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    const int k = j * S;
    val = exp(math::utils::LDGC<AccT>(offset_x + k) - warp_max);
    offset_y[k] = val / warp_sum;
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
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    val = max(val, math::utils::LDGC<AccT>(offset_x + j * S));
  }
  const AccT warp_max = WarpAllReduce<AccT, cub::Max, kWarpSize>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    val += exp(math::utils::LDGC<AccT>(offset_x + j * S) - warp_max);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, kWarpSize>(val);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    const int k = j * S;
    val = math::utils::LDGC<AccT>(offset_x + k) - warp_max;
    offset_y[k] = val - log(warp_sum);
  }
}

template <typename T, typename AccT>
__global__ void _BlockSoftmax(const int S, const int C, const T* x, T* y) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    val = max(val, math::utils::LDGC<AccT>(offset_x + j * S));
  }
  const AccT block_max = BlockAllReduce<AccT, cub::Max, kBlockSize>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    val += exp(math::utils::LDGC<AccT>(offset_x + j * S) - block_max);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, kBlockSize>(val);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    const int k = j * S;
    val = exp(math::utils::LDGC<AccT>(offset_x + k) - block_max);
    offset_y[k] = val / block_sum;
  }
}

template <typename T, typename AccT>
__global__ void _BlockLogSoftmax(const int S, const int C, const T* x, T* y) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto* offset_x = x + offset;
  auto* offset_y = y + offset;
  AccT val = AccT(-FLT_MAX);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    val = max(val, math::utils::LDGC<AccT>(offset_x + j * S));
  }
  const AccT block_max = BlockAllReduce<AccT, cub::Max, kBlockSize>(val);
  val = AccT(0);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    val += exp(math::utils::LDGC<AccT>(offset_x + j * S) - block_max);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, kBlockSize>(val);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    const int k = j * S;
    val = math::utils::LDGC<AccT>(offset_x + k) - block_max;
    offset_y[k] = val - log(block_sum);
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
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    const int k = j * S;
    val += math::utils::LDGC<AccT>(offset_dy + k) *
        math::utils::LDGC<AccT>(offset_y + k);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, kWarpSize>(val);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    const int k = j * S;
    val = math::utils::LDGC<AccT>(offset_dy + k) - warp_sum;
    offset_dx[k] = val * math::utils::LDGC<AccT>(offset_y + k);
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
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    val += math::utils::LDGC<AccT>(offset_dy + j * S);
  }
  const AccT warp_sum = WarpAllReduce<AccT, cub::Sum, kWarpSize>(val);
  for (int j = threadIdx.x; j < C; j += kWarpSize) {
    const int k = j * S;
    val = exp(convert::To<AccT>(offset_y[k])) * warp_sum;
    offset_dx[k] = math::utils::LDGC<AccT>(offset_dy + k) - val;
  }
}

template <typename T, typename AccT>
__global__ void
_BlockSoftmaxGrad(const int S, const int C, const T* dy, const T* y, T* dx) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto *offset_dy = dy + offset, *offset_y = y + offset;
  auto* offset_dx = dx + offset;
  AccT val = AccT(0);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    const int k = j * S;
    val += math::utils::LDGC<AccT>(offset_dy + k) *
        math::utils::LDGC<AccT>(offset_y + k);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, kBlockSize>(val);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    const int k = j * S;
    val = math::utils::LDGC<AccT>(offset_dy + k) - block_sum;
    offset_dx[k] = val * math::utils::LDGC<AccT>(offset_y + k);
  }
}

template <typename T, typename AccT>
__global__ void
_BlockLogSoftmaxGrad(const int S, const int C, const T* dy, const T* y, T* dx) {
  const int offset = (blockIdx.x / S) * C * S + (blockIdx.x % S);
  auto *offset_dy = dy + offset, *offset_y = y + offset;
  auto* offset_dx = dx + offset;
  AccT val = AccT(0);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    val += math::utils::LDGC<AccT>(offset_dy + j * S);
  }
  const AccT block_sum = BlockAllReduce<AccT, cub::Sum, kBlockSize>(val);
  for (int j = threadIdx.x; j < C; j += kBlockSize) {
    const int k = j * S;
    val = exp(convert::To<AccT>(offset_y[k])) * block_sum;
    offset_dx[k] = math::utils::LDGC<AccT>(offset_dy + k) - val;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                        \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int N,                                                             \
      const int S,                                                             \
      const int C,                                                             \
      const T* x,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    using ScalarT = math::Traits<T>::scalar_type;                              \
    using AccT = math::Traits<T>::accumulator_type;                            \
    const auto NxS = N * S;                                                    \
    if (C <= 1024) {                                                           \
      const auto num_threads = dim3(kWarpSize, std::min(NxS, kWarpItems));     \
      const auto num_blocks = math::utils::DivUp<int>(NxS, num_threads.y);     \
      _Warp##name<ScalarT, AccT>                                               \
          <<<num_blocks, num_threads, 0, ctx->cuda_stream()>>>(                \
              NxS, S, C, (const ScalarT*)x, (ScalarT*)y);                      \
    } else {                                                                   \
      _Block##name<ScalarT, AccT><<<NxS, kBlockSize, 0, ctx->cuda_stream()>>>( \
          S, C, (const ScalarT*)x, (ScalarT*)y);                               \
    }                                                                          \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                                   \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int N,                                                             \
      const int S,                                                             \
      const int C,                                                             \
      const T* dy,                                                             \
      const T* y,                                                              \
      T* dx,                                                                   \
      CUDAContext* ctx) {                                                      \
    using ScalarT = math::Traits<T>::scalar_type;                              \
    using AccT = math::Traits<T>::accumulator_type;                            \
    const auto NxS = N * S;                                                    \
    if (C <= 1024) {                                                           \
      const auto num_threads = dim3(kWarpSize, std::min(NxS, kWarpItems));     \
      const auto num_blocks = math::utils::DivUp<int>(NxS, num_threads.y);     \
      _Warp##name<ScalarT, AccT>                                               \
          <<<num_blocks, num_threads, 0, ctx->cuda_stream()>>>(                \
              NxS, S, C, (const ScalarT*)dy, (const ScalarT*)y, (ScalarT*)dx); \
    } else {                                                                   \
      _Block##name<ScalarT, AccT><<<NxS, kBlockSize, 0, ctx->cuda_stream()>>>( \
          S, C, (const ScalarT*)dy, (const ScalarT*)y, (ScalarT*)dx);          \
    }                                                                          \
  }

DEFINE_KERNEL_LAUNCHER(Softmax, float16);
DEFINE_KERNEL_LAUNCHER(Softmax, bfloat16);
DEFINE_KERNEL_LAUNCHER(Softmax, float);
DEFINE_KERNEL_LAUNCHER(Softmax, double);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, bfloat16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
