#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! DepthwiseConv2d <T = float32, Device = CUDA> */

template <typename T, int KKH, int KKW>
__global__ void _DepthwiseConv2d_NCHW(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    const T*                w,
    T*                      y) {
    const int KH = KKH < 0 ? kernel_h : KKH;
    const int KW = KKW < 0 ? kernel_w : KKW;
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int OW = idx % out_w;
        const int OH = (idx / out_w) % out_h;
        const int OC = (idx / out_w / out_h) % C;
        const int OB = idx / out_w / out_h / C;

        const int ih_start = OH * stride - pad_h;
        const int iw_start = OW * stride - pad_w;
        const int ih_end = ih_start + KH;
        const int iw_end = iw_start + KW;

        const int fc_start = OC * KH * KW;
        const int xc_start = (OB * C + OC) * H * W;

        T sum = 0;
        if (ih_start >= 0 && iw_start >= 0 &&
                ih_end < H && iw_end < W) {
            // Loop that doesn't need to check for boundary conditions
#pragma unroll
            for (int fh = 0; fh < KH; ++fh) {
                const int ih = ih_start + fh;
                const int x_start = xc_start + ih * W;
                const int f_start = fc_start + fh * KW;
#pragma unroll
                for (int fw = 0; fw < KW; ++fw) {
                    const int iw = iw_start + fw;
#if __CUDA_ARCH__ >= 350
                    sum += __ldg(x + x_start + iw) * __ldg(w + f_start + fw);
#else
                    sum += x[x_start + iw] * w[f_start + fw];
#endif
                }  // End fw
            } // End fh
        } else {
            // Loop that needs to check for boundary conditions
#pragma unroll
            for (int fh = 0; fh < KH; ++fh) {
                const int ih = ih_start + fh;
                const int x_start = xc_start + ih * W;
                const int f_start = fc_start + fh * KW;
#pragma unroll
                for (int fw = 0; fw < KW; ++fw) {
                    const int iw = iw_start + fw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
#if __CUDA_ARCH__ >= 350
                    sum += __ldg(x + x_start + iw) * __ldg(w + f_start + fw);
#else
                    sum += x[x_start + iw] * w[f_start + fw];
#endif
                    }
                }  // End fw
            }  // End fh
        }
        y[idx] = sum;
    }
}

template <typename T, int KKH, int KKW>
__global__ void _DepthwiseConv2d_NHWC(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    const T*                w,
    T*                      y) {
    const int KH = KKH < 0 ? kernel_h : KKH;
    const int KW = KKW < 0 ? kernel_w : KKW;
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int OC = idx % C;
        const int OW = (idx / C) % out_w;
        const int OH = (idx / C / out_w) % out_h;
        const int OB = idx / C / out_h / out_h;

        const int ih_start = OH * stride - pad_h;
        const int iw_start = OW * stride - pad_w;
        const int ih_end = ih_start + KH;
        const int iw_end = iw_start + KW;

        const int xb_start = OB * H;
        const int fc_start = OC  * KH;

        T sum = 0;

        if (ih_start >= 0 && iw_start >= 0 &&
                ih_end < H && iw_end < W) {
            // Loop that doesn't need to check for boundary conditions
#pragma unroll
            for (int fh = 0; fh < KH; ++fh) {
                const int ih = ih_start + fh;
                const int x_start = (xb_start + ih) * W;
                const int f_start = (fc_start + fh) * KW;
#pragma unroll
                for (int fw = 0; fw < KW; ++fw) {
                    const int iw = iw_start + fw;
                    const int x_idx = (x_start + iw) * C + OC;
#if __CUDA_ARCH__ >= 350
                    sum += __ldg(x + x_idx) * __ldg(w + f_start + fw);
#else
                    sum += x[x_idx] * w[f_start + fw];
#endif
                }
            }
        } else {
#pragma unroll
            for (int fh = 0; fh < KH; ++fh) {
                const int ih = ih_start + fh;
                const int x_start = (xb_start + ih) * W;
                const int f_start = (fc_start + fh) * KW;
#pragma unroll
                for (int fw = 0; fw < KW; ++fw) {
                    const int iw = iw_start + fw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        const int x_idx = (x_start + iw) * C + OC;
#if __CUDA_ARCH__ >= 350
                        sum += __ldg(x + x_idx) * __ldg(w + f_start + fw);
#else
                        sum += x[x_idx] * w[f_start + fw];
#endif
                    }
                }  // End fw
            }  // End fh
        }
        y[idx] = sum;
    }
}

template <> void DepthwiseConv2d<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    const float*            w,
    float*                  y,
    CUDAContext*            ctx) {
    const auto count = N * C * out_h * out_w;
    if (data_format == "NCHW") {
        if (kernel_h == 3 && kernel_w == 3) {
            _DepthwiseConv2d_NCHW<float, 3, 3>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, x, w, y);
        } else {
            _DepthwiseConv2d_NCHW<float, -1, -1>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, x, w, y);
        }
    } else if (data_format == "NHWC") {
        if (kernel_h == 3 && kernel_w == 3) {
            _DepthwiseConv2d_NHWC<float, 3, 3>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, x, w, y);
        } else {
            _DepthwiseConv2d_NHWC<float, -1, -1>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, x, w, y);
        }
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T, int KKH, int KKW>
__global__ void _DepthwiseConv2dGrad_NCHW(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const T*                w,
    T*                      dx) { 
    const int KH = KKH < 0 ? kernel_h : KKH;
    const int KW = KKW < 0 ? kernel_w : KKW;
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int IW = idx % W;
        const int IH = (idx / W) % H;
        const int IC = (idx / W / H) % C;
        const int IB = idx / W / H / C;

        const int oh_start = max(0, (IH - KH + pad_h + stride) / stride);
        const int oh_end = min(out_h - 1, (IH + pad_h) / stride);
        const int ow_start = max(0, (IW - KW + pad_w + stride) / stride);
        const int ow_end = min(out_w - 1, (IW + pad_w) / stride);

        const int fc_start = IC * KH * KW;
        const int yc_start = (IB * C + IC) * (out_h * out_w);

        T sum = 0;
#pragma unroll
        for (int oh = oh_start; oh <= oh_end; ++oh) {
            const int fh = IH + pad_h - oh * stride;
            const int f_start = fc_start + fh * KW;
            const int y_start = yc_start + oh * out_w;
            for (int ow = ow_start; ow <= ow_end; ++ow) {
                const int fw = IW + pad_w - ow * stride;
#if __CUDA_ARCH__ >= 350
                sum += __ldg(dy + y_start + ow) * __ldg(w + f_start + fw);
#else
                sum += dy[y_start + ow] * w[f_start + fw];
#endif
            }  // End yw
        }  // End yh
        dx[idx] = sum;
  }
}

template <> void DepthwiseConv2dGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    const float*            w,
    float*                  dx,
    CUDAContext*            ctx) {
    const auto count = N * C * out_h * out_w;
    if (data_format == "NCHW") {
        if (kernel_h == 3 && kernel_w == 3) {
            _DepthwiseConv2dGrad_NCHW<float, 3, 3>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, dy, w, dx);
        } else {
            _DepthwiseConv2dGrad_NCHW<float, -1, -1>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, dy, w, dx);
        }
    } else if (data_format == "NHWC") {
        if (kernel_h == 3 && kernel_w == 3) {
            NOT_IMPLEMENTED;
        } else {
            NOT_IMPLEMENTED;
        }
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! DepthwiseConv2dWGrad <T = float32, Device = CUDA> */

template <typename T, int KKH, int KKW>
__global__ void _DepthwiseConv2dWGrad_NCHW(
    const int               count,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const T*                x,
    T*                      dw) {
    const int KH = KKH < 0 ? kernel_h : KKH;
    const int KW = KKW < 0 ? kernel_w : KKW;
    CUDA_1D_KERNEL_LOOP(idx, count) {
        const int OW = idx % out_w;
        const int OH = (idx / out_w) % out_h;
        const int OC = (idx / out_w / out_h) % C;
        const int OB = idx / out_w / out_h / C;

        const int ih_start = OH * stride - pad_h;
        const int iw_start = OW * stride - pad_w;
        const int ih_end = ih_start + KH;
        const int iw_end = iw_start + KW;

        const int fc_start = OC * KH * KW;
        const int xc_start = (OB * C + OC) * (H * W);

#if __CUDA_ARCH__ >= 350
        const T dY = __ldg(dy + idx);
#else
        const T dY = dy[idx];
#endif
        if (ih_start >= 0 && iw_start >= 0 &&
                ih_end < H && iw_end < W) {
#pragma unroll
            for (int fh = 0; fh < KH; ++fh) {
                const int ih = ih_start + fh;
                const int x_start = xc_start + ih * W;
                const int f_start = fc_start + fh * KW;
#pragma unroll
            for (int fw = 0; fw < KW; ++fw) {
                const int iw = iw_start + fw;
#if __CUDA_ARCH__ >= 350
                T partial_sum = __ldg(x + x_start + iw) * dY;
#else
                T partial_sum = x[x_start + iw] * dY;
#endif
                atomicAdd(dw + f_start + fw, partial_sum);
            }  // End fw
        }  // End fh
    } else {
#pragma unroll
            for (int fh = 0; fh < KH; ++fh) {
                const int ih = ih_start + fh;
                const int x_start = xc_start + ih * W;
                const int f_start = fc_start + fh * KW;
#pragma unroll
                for (int fw = 0; fw < KW; ++fw) {
                    const int iw = iw_start + fw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
#if __CUDA_ARCH__ >= 350
                        T partial_sum = __ldg(x + x_start + iw) * dY;
#else
                        T partial_sum = x[x_start + iw] * dY;
#endif
                        atomicAdd(dw + f_start + fw, partial_sum);
                    } // End if
                }  // End fw
            }  // End fh
        }  // End if
    }
}

template <> void DepthwiseConv2dWGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    float*                  dw,
    CUDAContext*            ctx) {
    const auto count = N * C * out_h * out_w;
    if (data_format == "NCHW") {
        if (kernel_h == 3 && kernel_w == 3) {
            _DepthwiseConv2dWGrad_NCHW<float, 3, 3>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, dy, x, dw);
        } else {
            _DepthwiseConv2dWGrad_NCHW<float, -1, -1>
                << < CUDA_BLOCKS(count), CUDA_THREADS,
                     0, ctx->cuda_stream() >> >
                (count, C, H, W, out_h, out_w,
                    kernel_h, kernel_w, stride,
                        pad_h, pad_w, dy, x, dw);
        }
    } else if (data_format == "NHWC") {
        if (kernel_h == 3 && kernel_w == 3) {
            NOT_IMPLEMENTED;
        }
        else {
            NOT_IMPLEMENTED;
        }
    }
    else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA