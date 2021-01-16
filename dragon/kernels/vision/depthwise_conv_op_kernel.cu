#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#else
#define LDG(x, i) x[i]
#endif

template <typename T, typename AccT, int kPatchH, int kPatchW>
__global__ void _DepthwiseConv2dNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const T* x,
    const T* filter,
    T* y) {
  const int patch_h = kPatchH < 0 ? kernel_h : kPatchH;
  const int patch_w = kPatchW < 0 ? kernel_w : kPatchW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const int hstart = h_out * stride_h - pad_h;
    const int wstart = w_out * stride_w - pad_w;
    const int x_offset = (n * C + c) * H * W;
    int fi = c * patch_h * patch_w;

    AccT val = AccT(0);
#pragma unroll
    for (int h_k = 0; h_k < patch_h; ++h_k) {
#pragma unroll
      for (int w_k = 0; w_k < patch_w; ++w_k) {
        const int h = hstart + h_k * dilation_h;
        const int w = wstart + w_k * dilation_w;
        if (math::utils::IsAGeZeroAndALtB(h, H) &&
            math::utils::IsAGeZeroAndALtB(w, W)) {
          const int xi = x_offset + h * W + w;
          val += convert::To<AccT>(Multiplies(LDG(x, xi), LDG(filter, fi)));
        }
        ++fi;
      }
    }
    y[yi] = convert::To<T>(val);
  }
}

template <typename T, typename AccT, int kPatchH, int kPatchW>
__global__ void _DepthwiseConv2dNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const T* x,
    const T* filter,
    T* y) {
  const int patch_h = kPatchH < 0 ? kernel_h : kPatchH;
  const int patch_w = kPatchW < 0 ? kernel_w : kPatchW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w_out = (yi / C) % out_w;
    const int h_out = (yi / C / out_w) % out_h;
    const int n = yi / C / out_h / out_h;

    const int hstart = h_out * stride_h - pad_h;
    const int wstart = w_out * stride_w - pad_w;
    const int x_offset = n * H * W * C + c;
    int fi = c * patch_h * patch_w;

    AccT val = AccT(0);
#pragma unroll
    for (int h_k = 0; h_k < patch_h; ++h_k) {
#pragma unroll
      for (int w_k = 0; w_k < patch_w; ++w_k) {
        const int h = hstart + h_k * dilation_h;
        const int w = wstart + w_k * dilation_w;
        if (math::utils::IsAGeZeroAndALtB(h, H) &&
            math::utils::IsAGeZeroAndALtB(w, W)) {
          const int xi = x_offset + (h * W + w) * C;
          val += convert::To<AccT>(Multiplies(LDG(x, xi), LDG(filter, fi)));
        }
        ++fi;
      }
    }
    y[yi] = convert::To<T>(val);
  }
}

template <typename T, typename AccT, int kPatchH, int kPatchW>
__global__ void _DepthwiseConv2dGradNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const T* dy,
    const T* filter,
    T* dx) {
  const int patch_h = kPatchH < 0 ? kernel_h : kPatchH;
  const int patch_w = kPatchW < 0 ? kernel_w : kPatchW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int w = xi % W;
    const int h = (xi / W) % H;
    const int c = (xi / W / H) % C;
    const int n = xi / W / H / C;

    const int y_offset = (n * C + c) * out_h * out_w;
    int fi = c * patch_h * patch_w;

    AccT val = AccT(0);
#pragma unroll
    for (int h_k = 0; h_k < patch_h; ++h_k) {
#pragma unroll
      for (int w_k = 0; w_k < patch_w; ++w_k) {
        int h_out = h + pad_h - h_k * dilation_h;
        int w_out = w + pad_w - w_k * dilation_w;
        if ((h_out % stride_h == 0) && (w_out % stride_w == 0)) {
          h_out = h_out / stride_h;
          w_out = w_out / stride_w;
          if (math::utils::IsAGeZeroAndALtB(h_out, out_h) &&
              math::utils::IsAGeZeroAndALtB(w_out, out_w)) {
            const int yi = y_offset + h_out * out_w + w_out;
            val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(filter, fi)));
          }
        }
        ++fi;
      }
    }
    dx[xi] = convert::To<T>(val);
  }
}

template <typename T, typename AccT, int kPatchH, int kPatchW>
__global__ void _DepthwiseConv2dGradNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const T* dy,
    const T* filter,
    T* dx) {
  const int patch_h = kPatchH < 0 ? kernel_h : kPatchH;
  const int patch_w = kPatchW < 0 ? kernel_w : kPatchW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int c = xi % C;
    const int w = (xi / C) % W;
    const int h = (xi / C / W) % H;
    const int n = xi / C / W / H;

    const int y_offset = n * out_h * out_w * C + c;
    int fi = c * patch_h * patch_w;

    AccT val = AccT(0);
#pragma unroll
    for (int h_k = 0; h_k < patch_h; ++h_k) {
#pragma unroll
      for (int w_k = 0; w_k < patch_w; ++w_k) {
        int h_out = h + pad_h - h_k * dilation_h;
        int w_out = w + pad_w - w_k * dilation_w;
        if ((h_out % stride_h == 0) && (w_out % stride_w == 0)) {
          h_out = h_out / stride_h;
          w_out = w_out / stride_w;
          if (math::utils::IsAGeZeroAndALtB(h_out, out_h) &&
              math::utils::IsAGeZeroAndALtB(w_out, out_w)) {
            const int yi = y_offset + (h_out * out_w + w_out) * C;
            val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(filter, fi)));
          }
        }
        ++fi;
      }
    }
    dx[xi] = val;
  }
}

template <typename T, typename AccT>
__global__ void _DepthwiseConv2dWGradNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const T* dy,
    const T* x,
    T* dfilter) {
  const auto Multiplies = math::MultipliesFunctor<T>();
  const int block_idx = blockIdx.x;
  const int w_k = block_idx % kernel_w;
  const int h_k = (block_idx / kernel_w) % kernel_h;
  const int c = block_idx / kernel_w / kernel_h;

  const int n = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int lane_idx = threadIdx.x % 32;
  const int out_dim = out_h * out_w;

  AccT val = AccT(0);
  for (int i = n; i < N; i += nwarps) {
    for (int j = lane_idx; j < out_dim; j += 32) {
      const int h = (j / out_w) * stride_h - pad_h + h_k * dilation_h;
      const int w = (j % out_w) * stride_w - pad_w + w_k * dilation_w;
      if (math::utils::IsAGeZeroAndALtB(h, H) &&
          math::utils::IsAGeZeroAndALtB(w, W)) {
        const int xi = ((i * C + c) * H + h) * W + w;
        const int yi = (i * C + c) * out_dim + j;
        val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(x, xi)));
      }
    }
  }

  typedef cub::BlockReduce<AccT, 256> Reduce;
  __shared__ typename Reduce::TempStorage storage;
  val = Reduce(storage).Sum(val);
  if (threadIdx.x == 0) {
    dfilter[block_idx] = convert::To<T>(val);
  }
}

template <typename T, typename AccT>
__global__ void _DepthwiseConv2dWGradNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const T* dy,
    const T* x,
    T* dfilter) {
  const auto Multiplies = math::MultipliesFunctor<T>();
  const int block_idx = blockIdx.x;
  const int w_k = block_idx % kernel_w;
  const int h_k = (block_idx / kernel_w) % kernel_h;
  const int c = block_idx / kernel_w / kernel_h;

  const int n = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int lane_idx = threadIdx.x % 32;
  const int out_dim = out_h * out_w;

  AccT val = AccT(0);
  for (int i = n; i < N; i += nwarps) {
    for (int j = lane_idx; j < out_dim; j += 32) {
      const int h = (j / out_w) * stride_h - pad_h + h_k * dilation_h;
      const int w = (j % out_w) * stride_w - pad_w + w_k * dilation_w;
      if (math::utils::IsAGeZeroAndALtB(h, H) &&
          math::utils::IsAGeZeroAndALtB(w, W)) {
        const int xi = ((i * H + h) * W + w) * C + c;
        const int yi = (i * out_dim + j) * C + c;
        val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(x, xi)));
      }
    }
  }

  typedef cub::BlockReduce<AccT, 256> Reduce;
  __shared__ typename Reduce::TempStorage storage;
  val = Reduce(storage).Sum(val);
  if (threadIdx.x == 0) {
    dfilter[block_idx] = convert::To<T>(val);
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_DATA_KERNEL(name, T, AccT, nblocks, nthreads, ...)    \
  if (data_format == "NCHW") {                                         \
    if (kernel_h == 3 && kernel_w == 3) {                              \
      name##NCHW<T, AccT, 3, 3>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 5 && kernel_w == 5) {                       \
      name##NCHW<T, AccT, 5, 5>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 7 && kernel_w == 7) {                       \
      name##NCHW<T, AccT, 7, 7>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else {                                                           \
      name##NCHW<T, AccT, -1, -1>                                      \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    }                                                                  \
  } else if (data_format == "NHWC") {                                  \
    if (kernel_h == 3 && kernel_w == 3) {                              \
      name##NHWC<T, AccT, 3, 3>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 5 && kernel_w == 5) {                       \
      name##NHWC<T, AccT, 5, 5>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 7 && kernel_w == 7) {                       \
      name##NHWC<T, AccT, 7, 7>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else {                                                           \
      name##NHWC<T, AccT, -1, -1>                                      \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    }                                                                  \
  } else {                                                             \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;               \
  }

#define DISPATCH_FILTER_KERNEL(name, T, AccT, nblocks, nthreads, ...) \
  if (data_format == "NCHW") {                                        \
    name##NCHW<T, AccT>                                               \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);  \
  } else if (data_format == "NHWC") {                                 \
    name##NHWC<T, AccT>                                               \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);  \
  } else {                                                            \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;              \
  }

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                             \
  template <>                                                       \
  void DepthwiseConv2d<T, CUDAContext>(                             \
      const int N,                                                  \
      const int C,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_h,                                              \
      const int out_w,                                              \
      const int kernel_h,                                           \
      const int kernel_w,                                           \
      const int stride_h,                                           \
      const int stride_w,                                           \
      const int pad_h,                                              \
      const int pad_w,                                              \
      const int dilation_h,                                         \
      const int dilation_w,                                         \
      const string& data_format,                                    \
      const T* x,                                                   \
      const T* filter,                                              \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    const auto nthreads = N * C * out_h * out_w;                    \
    DISPATCH_DATA_KERNEL(                                           \
        _DepthwiseConv2d,                                           \
        math::ScalarType<T>::type,                                  \
        AccT,                                                       \
        CUDA_BLOCKS(nthreads),                                      \
        CUDA_THREADS,                                               \
        nthreads,                                                   \
        C,                                                          \
        H,                                                          \
        W,                                                          \
        out_h,                                                      \
        out_w,                                                      \
        kernel_h,                                                   \
        kernel_w,                                                   \
        stride_h,                                                   \
        stride_w,                                                   \
        pad_h,                                                      \
        pad_w,                                                      \
        dilation_h,                                                 \
        dilation_w,                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),      \
        reinterpret_cast<const math::ScalarType<T>::type*>(filter), \
        reinterpret_cast<math::ScalarType<T>::type*>(y));           \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, AccT)                        \
  template <>                                                       \
  void DepthwiseConv2dGrad<T, CUDAContext>(                         \
      const int N,                                                  \
      const int C,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_h,                                              \
      const int out_w,                                              \
      const int kernel_h,                                           \
      const int kernel_w,                                           \
      const int stride_h,                                           \
      const int stride_w,                                           \
      const int pad_h,                                              \
      const int pad_w,                                              \
      const int dilation_h,                                         \
      const int dilation_w,                                         \
      const string& data_format,                                    \
      const T* dy,                                                  \
      const T* filter,                                              \
      T* dx,                                                        \
      CUDAContext* ctx) {                                           \
    auto nthreads = N * C * H * W;                                  \
    DISPATCH_DATA_KERNEL(                                           \
        _DepthwiseConv2dGrad,                                       \
        math::ScalarType<T>::type,                                  \
        AccT,                                                       \
        CUDA_BLOCKS(nthreads),                                      \
        CUDA_THREADS,                                               \
        nthreads,                                                   \
        C,                                                          \
        H,                                                          \
        W,                                                          \
        out_h,                                                      \
        out_w,                                                      \
        kernel_h,                                                   \
        kernel_w,                                                   \
        stride_h,                                                   \
        stride_w,                                                   \
        pad_h,                                                      \
        pad_w,                                                      \
        dilation_h,                                                 \
        dilation_w,                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),     \
        reinterpret_cast<const math::ScalarType<T>::type*>(filter), \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));          \
  }                                                                 \
  template <>                                                       \
  void DepthwiseConv2dWGrad<T, CUDAContext>(                        \
      const int N,                                                  \
      const int C,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_h,                                              \
      const int out_w,                                              \
      const int kernel_h,                                           \
      const int kernel_w,                                           \
      const int stride_h,                                           \
      const int stride_w,                                           \
      const int pad_h,                                              \
      const int pad_w,                                              \
      const int dilation_h,                                         \
      const int dilation_w,                                         \
      const string& data_format,                                    \
      const T* dy,                                                  \
      const T* x,                                                   \
      T* dfilter,                                                   \
      CUDAContext* ctx) {                                           \
    const auto nblocks = C * kernel_h * kernel_w;                   \
    const auto nthreads = 256;                                      \
    DISPATCH_FILTER_KERNEL(                                         \
        _DepthwiseConv2dWGrad,                                      \
        math::ScalarType<T>::type,                                  \
        AccT,                                                       \
        nblocks,                                                    \
        nthreads,                                                   \
        N,                                                          \
        C,                                                          \
        H,                                                          \
        W,                                                          \
        out_h,                                                      \
        out_w,                                                      \
        kernel_h,                                                   \
        kernel_w,                                                   \
        stride_h,                                                   \
        stride_w,                                                   \
        pad_h,                                                      \
        pad_w,                                                      \
        dilation_h,                                                 \
        dilation_w,                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),     \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),      \
        reinterpret_cast<math::ScalarType<T>::type*>(dfilter));     \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_DATA_KERNEL
#undef DISPATCH_FILTER_KERNEL

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
