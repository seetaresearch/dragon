#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

constexpr int kTileDim = 32;
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

template <typename T>
__global__ void _BatchTranspose2D(
    const int H,
    const int W,
    const int dh,
    const int dw,
    const T* X,
    T* Y) {
  __shared__ T block[kTileDim][kTileDim + 1];
  const int k = blockIdx.x % (dh * dw);
  const int r = k / dw;
  const int c = k % dw;
  const int offset = blockIdx.x / (dh * dw) * H * W;
  int x = c * kTileDim + threadIdx.x;
  int y = r * kTileDim + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
      block[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x;
  y = c * kTileDim + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
      Y[offset + (y + i) * H + x] = block[threadIdx.x][threadIdx.y + i];
    }
  }
}

template <typename T, int D>
void _TransposeImpl(
    const int64_t* x_strides,
    const int64_t* y_dims,
    const T* x,
    T* y,
    CUDAContext* ctx) {
  const auto N =
      std::accumulate(y_dims, y_dims + D, 1, std::multiplies<int64_t>());
  SimpleArray<int, D> X_strides, Y_dims;
  for (int i = 0; i < D; ++i) {
    X_strides.data[i] = x_strides[i];
    Y_dims.data[i] = y_dims[i];
  }
  _Transpose<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, X_strides, Y_dims, x, y);
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
      const auto N = new_dims[0], H = new_dims[1], W = new_dims[2];         \
      const auto dh = utils::DivUp<int64_t>(H, kTileDim);                   \
      const auto dw = utils::DivUp<int64_t>(W, kTileDim);                   \
      _BatchTranspose2D<<<                                                  \
          N * dh * dw,                                                      \
          dim3(kTileDim, kBlockRows),                                       \
          0,                                                                \
          ctx->cuda_stream()>>>(H, W, dh, dw, x, y);                        \
      return;                                                               \
    }                                                                       \
    CUDA_TENSOR_DIMS_CHECK(num_axes);                                       \
    vec64_t X_strides(num_axes), Y_dims(num_axes);                          \
    utils::ComputeTransposeStrides(                                         \
        num_axes, new_dims.data(), new_axes.data(), X_strides.data());      \
    for (int i = 0; i < num_axes; ++i) {                                    \
      Y_dims[i] = new_dims[new_axes[i]];                                    \
    }                                                                       \
    DISPATCH_FUNC_BY_VALUE_WITH_TYPE_1(                                     \
        _TransposeImpl,                                                     \
        T,                                                                  \
        num_axes,                                                           \
        X_strides.data(),                                                   \
        Y_dims.data(),                                                      \
        x,                                                                  \
        y,                                                                  \
        ctx);                                                               \
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
