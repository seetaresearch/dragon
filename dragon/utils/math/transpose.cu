#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

constexpr int kTileDim = 32;
constexpr int kAlignedTileDim = kTileDim * 2;
constexpr int kBlockRows = 8;

template <typename T, int D>
__global__ void _Transpose(
    const int N,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, N) {
    int xi = 0, tmp = yi;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[d], tmp, &tmp, &r);
      xi += r * X_strides.data[d];
    }
    y[yi] = x[xi];
  }
}

template <typename T, int D, size_t L>
void _AlignedTranspose(
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  const auto N = math::utils::Prod(D, Y_dims.data);
  using ScalarT = typename std::aligned_storage<L, L>::type;
  _Transpose<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N,
      X_strides,
      Y_dims,
      reinterpret_cast<const ScalarT*>(x),
      reinterpret_cast<ScalarT*>(y));
}

template <typename T>
__global__ void _BatchTranspose2D(
    const int H,
    const int W,
    const int dh,
    const int dw,
    const T* X,
    T* Y) {
  __shared__ T tile[kTileDim][kTileDim + 1];
  const int k = blockIdx.x % (dh * dw);
  const int r = k / dw;
  const int c = k % dw;
  const int offset = blockIdx.x / (dh * dw) * H * W;
  int x = c * kTileDim + threadIdx.x, y = r * kTileDim;
  if (x < W) {
#pragma unroll
    for (int i = threadIdx.y; i < kTileDim; i += kBlockRows) {
      if (y + i < H) {
        tile[i][threadIdx.x] = X[offset + (y + i) * W + x];
      }
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x, y = c * kTileDim;
  if (x < H) {
#pragma unroll
    for (int i = threadIdx.y; i < kTileDim; i += kBlockRows) {
      if (y + i < W) {
        Y[offset + (y + i) * H + x] = tile[threadIdx.x][i];
      }
    }
  }
}

template <typename T>
__global__ void _AlignedBatchTranspose2D(
    const int H,
    const int W,
    const int dh,
    const int dw,
    const T* X,
    T* Y) {
  using T2B = typename std::aligned_storage<2, 2>::type;
  using T4B = typename std::aligned_storage<4, 4>::type;
  const T4B* X2 = reinterpret_cast<const T4B*>(X);
  T4B* Y2 = reinterpret_cast<T4B*>(Y);
  __shared__ union {
    T2B data[kAlignedTileDim][kAlignedTileDim + 2];
    T4B data2[kAlignedTileDim][kAlignedTileDim / 2 + 1];
  } tile;
  const int k = blockIdx.x % (dh * dw);
  const int r = k / dw;
  const int c = k % dw;
  const int offset = blockIdx.x / (dh * dw) * H * W;
  int x = c * kAlignedTileDim + threadIdx.x * 2, y = r * kAlignedTileDim;
  if (x < W) {
#pragma unroll
    for (int i = threadIdx.y; i < kAlignedTileDim; i += kBlockRows) {
      if (y + i < H) {
        tile.data2[i][threadIdx.x] = X2[(offset + (y + i) * W + x) / 2];
      }
    }
  }
  __syncthreads();
  x = r * kAlignedTileDim + threadIdx.x * 2, y = c * kAlignedTileDim;
  if (x < H) {
#pragma unroll
    for (int i = threadIdx.y; i < kAlignedTileDim; i += kBlockRows) {
      if (y + i < W) {
        union {
          T2B data[2];
          T4B data2;
        } storage;
        storage.data[0] = tile.data[threadIdx.x * 2][i];
        storage.data[1] = tile.data[threadIdx.x * 2 + 1][i];
        Y2[(offset + (y + i) * H + x) / 2] = storage.data2;
      }
    }
  }
}

template <typename T, int D>
void _TransposeImpl(
    const vec64_t& dims,
    const vec64_t& axes,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  auto aligned_size = sizeof(T);
  if (axes.back() == D - 1) {
    aligned_size = utils::GetAlignedSize<T, 16>(dims[D - 1], x, y);
  }
  SimpleArray<int, D> X_dims, X_strides, Y_dims;
  for (int i = 0; i < D; ++i) {
    X_dims.data[i] = dims[i];
  }
  X_dims.data[D - 1] /= int(aligned_size / sizeof(T));
  utils::ComputeTransposeStrides(D, X_dims.data, axes.data(), X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = X_dims.data[axes[i]];
  }
  if (aligned_size == 1) {
    _AlignedTranspose<T, D, 1>(X_strides, Y_dims, x, y, ctx);
  } else if (aligned_size == 2) {
    _AlignedTranspose<T, D, 2>(X_strides, Y_dims, x, y, ctx);
  } else if (aligned_size == 4) {
    _AlignedTranspose<T, D, 4>(X_strides, Y_dims, x, y, ctx);
  } else if (aligned_size == 8) {
    _AlignedTranspose<T, D, 8>(X_strides, Y_dims, x, y, ctx);
  } else if (aligned_size == 16) {
    _AlignedTranspose<T, D, 16>(X_strides, Y_dims, x, y, ctx);
  } else {
    LOG(FATAL) << "Unsupported aligned size: " << aligned_size;
  }
}

template <typename T>
void _BatchTranspose2DImpl(
    const vec64_t& dims,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  const auto N = dims[0], H = dims[1], W = dims[2];
  if (sizeof(T) == 2 && H % 2 == 0 && W % 2 == 0) {
    if (utils::GetAlignedSize<T, 4>(2, x, y) == 4) {
      const auto dh = utils::DivUp<int64_t>(H, kAlignedTileDim);
      const auto dw = utils::DivUp<int64_t>(W, kAlignedTileDim);
      _AlignedBatchTranspose2D<<<
          N * dh * dw,
          dim3(kTileDim, kBlockRows),
          0,
          ctx->cuda_stream()>>>(H, W, dh, dw, x, y);
      return;
    }
  }
  const auto dh = utils::DivUp<int64_t>(H, kTileDim);
  const auto dw = utils::DivUp<int64_t>(W, kTileDim);
  _BatchTranspose2D<<<
      N * dh * dw,
      dim3(kTileDim, kBlockRows),
      0,
      ctx->cuda_stream()>>>(H, W, dh, dw, x, y);
}

} // namespace

#define DEFINE_TRANSPOSE_FUNC(T)                                            \
  template <>                                                               \
  DRAGON_API void Transpose<T, CUDAContext>(                                \
      const int num_dims,                                                   \
      const int64_t* dims,                                                  \
      const int64_t* axes,                                                  \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    vec64_t new_dims, new_axes;                                             \
    utils::CollapseTransposeAxes(num_dims, dims, axes, new_dims, new_axes); \
    const int num_axes = new_dims.size();                                   \
    if (num_axes == 3 && new_axes == vec64_t({0, 2, 1})) {                  \
      return _BatchTranspose2DImpl(new_dims, x, y, ctx);                    \
    }                                                                       \
    CUDA_TENSOR_DIMS_CHECK(num_axes);                                       \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1(                                     \
        _TransposeImpl, T, num_axes, new_dims, new_axes, x, y, ctx);        \
  }

DEFINE_TRANSPOSE_FUNC(bool);
DEFINE_TRANSPOSE_FUNC(uint8_t);
DEFINE_TRANSPOSE_FUNC(int8_t);
DEFINE_TRANSPOSE_FUNC(int);
DEFINE_TRANSPOSE_FUNC(int64_t);
DEFINE_TRANSPOSE_FUNC(float16);
DEFINE_TRANSPOSE_FUNC(float);
DEFINE_TRANSPOSE_FUNC(double);
#undef DEFINE_TRANSPOSE_FUNC

} // namespace math

} // namespace dragon

#endif // USE_CUDA
