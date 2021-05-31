#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Im2Col2dNCHW(
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
    const T* im,
    T* col) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int w_out = i % out_w;
    const int h_out = (i / out_w) % out_h;
    const int c = i / out_w / out_h;
    const int c_out = c * kernel_h * kernel_w;

    const int hstart = h_out * stride_h - pad_h;
    const int wstart = w_out * stride_w - pad_w;
    const T* offset_im = im + ((c * H + hstart) * W + wstart);
    T* offset_col = col + ((c_out * out_h + h_out) * out_w + w_out);

    for (int h_k = 0; h_k < kernel_h; h_k++) {
      for (int w_k = 0; w_k < kernel_w; w_k++) {
        const int h = hstart + h_k * dilation_h;
        const int w = wstart + w_k * dilation_w;
        *offset_col = (math::utils::IsAGeZeroAndALtB(h, H) &&
                       math::utils::IsAGeZeroAndALtB(w, W))
            ? __ldg(offset_im + h_k * dilation_h * W + w_k * dilation_w)
            : T(0);
        offset_col += (out_h * out_w);
      }
    }
  }
}

template <typename T>
__global__ void _Im2Col2dNHWC(
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
    const T* im,
    T* col) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int c = i % C;
    const int w_out = (i / C) % out_w;
    const int h_out = i / C / out_w;

    const int hstart = h_out * stride_h - pad_h;
    const int wstart = w_out * stride_w - pad_w;
    const int col_start = ((h_out * out_w) + w_out) * kernel_h;

    for (int h_k = 0; h_k < kernel_h; h_k++) {
      for (int w_k = 0; w_k < kernel_w; w_k++) {
        const int h = hstart + h_k * dilation_h;
        const int w = wstart + w_k * dilation_w;
        const int col_idx = (((col_start + h_k) * kernel_w + w_k) * C + c);
        col[col_idx] = (math::utils::IsAGeZeroAndALtB(h, H) &&
                        math::utils::IsAGeZeroAndALtB(w, W))
            ? __ldg(im + (h * W + w) * C + c)
            : T(0);
      }
    }
  }
}

template <typename T>
__global__ void _Col2Im2dNCHW(
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
    const T* col,
    T* im) {
  const int field_h = (kernel_h - 1) * dilation_h + 1;
  const int field_w = (kernel_w - 1) * dilation_w + 1;
  CUDA_1D_KERNEL_LOOP(im_idx, nthreads) {
    const int w = im_idx % W + pad_w;
    const int h = (im_idx / W) % H + pad_h;
    const int c = im_idx / W / H;

    const int out_hstart = (h < field_h) ? 0 : (h - field_h) / stride_h + 1;
    const int out_wstart = (w < field_w) ? 0 : (w - field_w) / stride_w + 1;
    const int out_hend = min(h / stride_h + 1, out_h);
    const int out_wend = min(w / stride_w + 1, out_w);

    float val = 0.f;
    int h_k, w_k, col_idx;
    for (int h_out = out_hstart; h_out < out_hend; ++h_out) {
      for (int w_out = out_wstart; w_out < out_wend; ++w_out) {
        h_k = (h - h_out * stride_h);
        w_k = (w - w_out * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          col_idx = (c * kernel_h + h_k) * kernel_w + w_k;
          col_idx = (col_idx * out_h + h_out) * out_w + w_out;
          val += convert::To<float>(__ldg(col + col_idx));
        }
      }
    }
    im[im_idx] = convert::To<T>(val);
  }
}

template <typename T>
__global__ void _Col2Im2dNHWC(
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
    const T* col,
    T* im) {
  const int field_h = (kernel_h - 1) * dilation_h + 1;
  const int field_w = (kernel_w - 1) * dilation_w + 1;
  CUDA_1D_KERNEL_LOOP(im_idx, nthreads) {
    const int c = im_idx % C;
    const int w = (im_idx / C) % W + pad_w;
    const int h = (im_idx / C / W) + pad_h;

    const int out_hstart = (h < field_h) ? 0 : (h - field_h) / stride_h + 1;
    const int out_wstart = (w < field_w) ? 0 : (w - field_w) / stride_w + 1;
    const int out_hend = min(h / stride_h + 1, out_h);
    const int out_wend = min(w / stride_w + 1, out_w);

    float val = 0.f;
    int h_k, w_k, col_idx;
    for (int h_out = out_hstart; h_out < out_hend; ++h_out) {
      for (int w_out = out_wstart; w_out < out_wend; ++w_out) {
        h_k = (h - h_out * stride_h);
        w_k = (w - w_out * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          col_idx = (h_out * out_w) + w_out;
          col_idx = ((col_idx * kernel_h) + h_k) * kernel_w + w_k;
          val += convert::To<float>(__ldg(col + col_idx * C + c));
        }
      }
    }
    im[im_idx] = convert::To<T>(val);
  }
}

template <typename T, int D, bool Transposed>
__global__ void _Im2ColNdNCHW(
    const int kernel_dim,
    const int outer_dim,
    const int inner_dim,
    const int num_dims,
    const SimpleArray<int, D> in_shape,
    const SimpleArray<int, D> out_shape,
    const SimpleArray<int, D> kshape,
    const SimpleArray<int, D> strides,
    const SimpleArray<int, D> pads,
    const SimpleArray<int, D> dilations,
    const T* x,
    T* y) {
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    int tmp = i;
    int kstarts[D];
#pragma unroll
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(kshape.data[d], tmp, &tmp, &r);
      kstarts[d] = -pads.data[d] + r * dilations.data[d];
    }
    CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
      tmp = j;
      int in_idx[D];
      bool is_padding = false;
#pragma unroll
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(out_shape.data[d], tmp, &tmp, &r);
        in_idx[d] = r = kstarts[d] + r * strides.data[d];
        is_padding |= !math::utils::IsAGeZeroAndALtB(r, in_shape.data[d]);
      }
      const int col_idx = i * inner_dim + j;
      int im_idx = i / kernel_dim;
#pragma unroll
      for (int d = 0; d < num_dims; ++d) {
        im_idx = im_idx * in_shape.data[d] + in_idx[d];
      }
      if (!Transposed) {
        y[col_idx] = is_padding ? T(0) : __ldg(x + im_idx);
      } else if (!is_padding) {
        math::utils::AtomicAdd(y + im_idx, x[col_idx]);
      }
    }
  }
}

template <typename T, int D, bool Transposed>
__global__ void _Im2ColNdNHWC(
    const int channel_dim,
    const int outer_dim,
    const int inner_dim,
    const int num_dims,
    const SimpleArray<int, D> in_shape,
    const SimpleArray<int, D> out_shape,
    const SimpleArray<int, D> kshape,
    const SimpleArray<int, D> strides,
    const SimpleArray<int, D> pads,
    const SimpleArray<int, D> dilations,
    const T* x,
    T* y) {
  CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
    int tmp = i / channel_dim;
    int kstarts[D];
#pragma unroll
    for (int d = num_dims - 1; d >= 0; --d) {
      int r;
      FIXED_DIVISOR_DIV_MOD(kshape.data[d], tmp, &tmp, &r);
      kstarts[d] = -pads.data[d] + r * dilations.data[d];
    }
    CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
      tmp = j;
      int in_idx[D];
      bool is_padding = false;
#pragma unroll
      for (int d = num_dims - 1; d >= 0; --d) {
        int r;
        FIXED_DIVISOR_DIV_MOD(out_shape.data[d], tmp, &tmp, &r);
        in_idx[d] = r = kstarts[d] + r * strides.data[d];
        is_padding |= !math::utils::IsAGeZeroAndALtB(r, in_shape.data[d]);
      }
      const int col_idx = j * outer_dim + i;
      int im_idx = in_idx[0];
#pragma unroll
      for (int d = 1; d < num_dims; ++d) {
        im_idx = im_idx * in_shape.data[d] + in_idx[d];
      }
      im_idx = im_idx * channel_dim + i % channel_dim;
      if (!Transposed) {
        y[col_idx] = is_padding ? convert::To<T>(0.f) : __ldg(x + im_idx);
      } else if (!is_padding) {
        math::utils::AtomicAdd(y + im_idx, x[col_idx]);
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_CONV_KERNEL(name, kBlocks, kThreads, ...)                 \
  if (data_format == "NCHW") {                                             \
    name##NCHW<<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else if (data_format == "NHWC") {                                      \
    name##NHWC<<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else {                                                                 \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                   \
  }

#define DEFINE_KERNEL_LAUNCHER(name, kTransposed, T)                   \
  template <>                                                          \
  void name<T, CUDAContext>(                                           \
      const int C,                                                     \
      const int H,                                                     \
      const int W,                                                     \
      const int out_h,                                                 \
      const int out_w,                                                 \
      const int kernel_h,                                              \
      const int kernel_w,                                              \
      const int stride_h,                                              \
      const int stride_w,                                              \
      const int pad_h,                                                 \
      const int pad_w,                                                 \
      const int dilation_h,                                            \
      const int dilation_w,                                            \
      const string& data_format,                                       \
      const T* x,                                                      \
      T* y,                                                            \
      CUDAContext* ctx) {                                              \
    const int nthreads = !kTransposed ? C * out_h * out_w : C * H * W; \
    DISPATCH_CONV_KERNEL(                                              \
        _##name,                                                       \
        CUDA_BLOCKS(nthreads),                                         \
        CUDA_THREADS,                                                  \
        nthreads,                                                      \
        C,                                                             \
        H,                                                             \
        W,                                                             \
        out_h,                                                         \
        out_w,                                                         \
        kernel_h,                                                      \
        kernel_w,                                                      \
        stride_h,                                                      \
        stride_w,                                                      \
        pad_h,                                                         \
        pad_w,                                                         \
        dilation_h,                                                    \
        dilation_w,                                                    \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),         \
        reinterpret_cast<math::ScalarType<T>::type*>(y));              \
  }

DEFINE_KERNEL_LAUNCHER(Im2Col2d, false, float16);
DEFINE_KERNEL_LAUNCHER(Im2Col2d, false, float);
DEFINE_KERNEL_LAUNCHER(Im2Col2d, false, double);
DEFINE_KERNEL_LAUNCHER(Col2Im2d, true, float16);
DEFINE_KERNEL_LAUNCHER(Col2Im2d, true, float);
DEFINE_KERNEL_LAUNCHER(Col2Im2d, true, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_CONV_KERNEL

#define DISPATCH_CONV_KERNEL(                                              \
    name, kTransposed, T, kBlocks, nthreads, channel_dim, kernel_dim, ...) \
  if (data_format == "NCHW") {                                             \
    name##NCHW<T, 3, kTransposed>                                          \
        <<<kBlocks, nthreads, 0, ctx->cuda_stream()>>>(                    \
            kernel_dim, __VA_ARGS__);                                      \
  } else if (data_format == "NHWC") {                                      \
    name##NHWC<T, 3, kTransposed>                                          \
        <<<kBlocks, nthreads, 0, ctx->cuda_stream()>>>(                    \
            channel_dim, __VA_ARGS__);                                     \
  } else {                                                                 \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                   \
  }

#define DEFINE_KERNEL_LAUNCHER(name, kTransposed, T)                           \
  template <>                                                                  \
  void name<T, CUDAContext>(                                                   \
      const int num_dims,                                                      \
      const int channels,                                                      \
      const int* in_shape,                                                     \
      const int* out_shape,                                                    \
      const int* kshape,                                                       \
      const int* strides,                                                      \
      const int* pads,                                                         \
      const int* dilations,                                                    \
      const string& data_format,                                               \
      const T* x,                                                              \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    CHECK_LE(num_dims, 3) << "Too many (> " << 3                               \
                          << ") dimensions to launch the cuda kernel.";        \
    SimpleArray<int, 3> in_shape_arr;                                          \
    SimpleArray<int, 3> out_shape_arr;                                         \
    SimpleArray<int, 3> kshape_arr;                                            \
    SimpleArray<int, 3> strides_arr;                                           \
    SimpleArray<int, 3> pads_arr;                                              \
    SimpleArray<int, 3> dilations_arr;                                         \
    for (int i = 0; i < num_dims; ++i) {                                       \
      in_shape_arr.data[i] = in_shape[i];                                      \
      out_shape_arr.data[i] = out_shape[i];                                    \
      kshape_arr.data[i] = kshape[i];                                          \
      strides_arr.data[i] = strides[i];                                        \
      pads_arr.data[i] = pads[i];                                              \
      dilations_arr.data[i] = dilations[i];                                    \
    }                                                                          \
    const auto kernel_dim =                                                    \
        std::accumulate(kshape, kshape + num_dims, 1, std::multiplies<int>()); \
    const auto inner_dim = std::accumulate(                                    \
        out_shape, out_shape + num_dims, 1, std::multiplies<int>());           \
    const auto outer_dim = channels * kernel_dim;                              \
    if (kTransposed) {                                                         \
      const auto in_dim = std::accumulate(                                     \
          in_shape, in_shape + num_dims, 1, std::multiplies<int>());           \
      math::Set(channels* in_dim, convert::To<T>(0.f), y, ctx);                \
    }                                                                          \
    DISPATCH_CONV_KERNEL(                                                      \
        _Im2ColNd,                                                             \
        kTransposed,                                                           \
        math::ScalarType<T>::type,                                             \
        outer_dim,                                                             \
        CUDA_THREADS,                                                          \
        channels,                                                              \
        kernel_dim,                                                            \
        outer_dim,                                                             \
        inner_dim,                                                             \
        num_dims,                                                              \
        in_shape_arr,                                                          \
        out_shape_arr,                                                         \
        kshape_arr,                                                            \
        strides_arr,                                                           \
        pads_arr,                                                              \
        dilations_arr,                                                         \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),                 \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                      \
  }

DEFINE_KERNEL_LAUNCHER(Im2ColNd, false, float16);
DEFINE_KERNEL_LAUNCHER(Im2ColNd, false, float);
DEFINE_KERNEL_LAUNCHER(Im2ColNd, false, double);
DEFINE_KERNEL_LAUNCHER(Col2ImNd, true, float16);
DEFINE_KERNEL_LAUNCHER(Col2ImNd, true, float);
DEFINE_KERNEL_LAUNCHER(Col2ImNd, true, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_CONV_KERNEL

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
