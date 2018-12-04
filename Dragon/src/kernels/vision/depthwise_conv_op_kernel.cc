#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! DepthwiseConv2d <T = float32, Device = CPU> */

template <typename T>
void _DepthwiseConv2d_NCHW(
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
    const T*                x,
    const T*                w,
    T*                      y) {
    for (int OB = 0; OB < N; ++OB) {
    for (int OC = 0; OC < C; ++OC) {
        const int yc_offset = OB * C + OC;
        const int xc_start = yc_offset * H * W;
        const int fc_start = OC * kernel_h * kernel_w;
        for (int OH = 0; OH < out_h; ++OH) {
            const int yh_offset = yc_offset * out_h + OH;
            const int ih_start = OH * stride - pad_h;
            const int ih_end = ih_start + kernel_h;
            for (int OW = 0; OW < out_w; ++OW) {
                T sum = 0;
                const int iw_start = OW * stride - pad_w;
                const int iw_end = iw_start + kernel_w;
                if (ih_start >= 0 && iw_start >= 0 &&
                        ih_end < H && iw_end < W) {
                    // Loop that doesn't need to check for boundary conditions.
                    for (int fh = 0; fh < kernel_h; ++fh) {
                        const int ih = ih_start + fh;
                        const int x_start = xc_start + ih * W;
                        const int f_start = fc_start + fh * kernel_w;
                        for (int fw = 0; fw < kernel_w; ++fw) {
                            const int iw = iw_start + fw;
                            sum += x[x_start + iw] * w[f_start + fw];
                        }  // End fw
                    } // End fh
                } else {
                    // Loop that needs to check for boundary conditions.
                    for (int fh = 0; fh < kernel_h; ++fh) {
                        const int ih = ih_start + fh;
                        const int x_start = xc_start + ih * W;
                        const int f_start = fc_start + fh * kernel_w;
                        for (int fw = 0; fw < kernel_w; ++fw) {
                            const int iw = iw_start + fw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += x[x_start + iw] * w[f_start + fw];
                            }
                        }  // End fw
                    }  // End fh
                }
                y[yh_offset * out_w + OW] = sum;
            }  // End OW
        }  // End OH
    }}  // End OC && OB
}

template <typename T>
void _DepthwiseConv2d_NHWC(
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
    const T*                x,
    const T*                w,
    T*                      y) {
    for (int OB = 0; OB < N; ++OB) {
        const int xb_start = OB * H;
        const int yb_start = OB * out_h;
        for (int OH = 0; OH < out_h; ++OH) {
            const int ih_start = OH * stride - pad_h;
            const int ih_end = ih_start + kernel_h;
            const int yh_start = (yb_start + OH) * out_w;
            for (int OW = 0; OW < out_w; ++OW) {
                const int iw_start = OW * stride - pad_w;
                const int iw_end = iw_start + kernel_w;
                const int yw_start = (yh_start + OW) * C;
                for (int OC = 0; OC < C; ++OC) {
                    const int fc_start = OC * kernel_h;
                    T sum = 0;
                    if (ih_start >= 0 && iw_start >= 0 &&
                            ih_end < H && iw_end < W) {
                        // Loop that doesn't need to check for boundary conditions.
                        for (int fh = 0; fh < kernel_h; ++fh) {
                            const int ih = ih_start + fh;
                            const int x_start = (xb_start + ih) * W;
                            const int f_start = (fc_start + fh) * kernel_w;
                            for (int fw = 0; fw < kernel_w; ++fw) {
                                const int iw = iw_start + fw;
                                const int x_idx = (x_start + iw) * C + OC;
                                sum += x[x_idx] * w[f_start + fw];
                            }  // End fw
                        } // End fh
                    } else {
                        // Loop that needs to check for boundary conditions.
                        for (int fh = 0; fh < kernel_h; ++fh) {
                            const int ih = ih_start + fh;
                            const int x_start = (xb_start + ih) * W;
                            const int f_start = (fc_start + fh) * kernel_w;
                            for (int fw = 0; fw < kernel_w; ++fw) {
                                const int iw = iw_start + fw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    const int x_idx = (x_start + iw) * C + OC;
                                    sum += x[x_idx] * w[f_start + fw];
                                }
                            }  // End fw
                        }  // End fh
                    }
                    y[yw_start + OC] = sum;
                }  // End OC
            }  // End OW
        }  // End OH
    }  // End OB
}

template <> void DepthwiseConv2d<float, CPUContext>(
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
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _DepthwiseConv2d_NCHW<float>(
            N, C, H, W, out_h, out_w,
                kernel_h, kernel_w, stride,
                    pad_h, pad_w, x, w, y);
    } else {
        _DepthwiseConv2d_NHWC<float>(
            N, C, H, W, out_h, out_w,
                kernel_h, kernel_w, stride,
                    pad_h, pad_w, x, w, y);
    }
}

template <> void DepthwiseConv2dGrad<float, CPUContext>(
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
    CPUContext*             ctx) {
    NOT_IMPLEMENTED;
}

template <> void DepthwiseConv2dWGrad<float, CPUContext>(
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
    CPUContext*             ctx) {
    NOT_IMPLEMENTED;
}

}  // namespace kernel

}  // namepsace dragon