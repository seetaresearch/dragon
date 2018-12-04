#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

bool less(int a, int b) { return unsigned(a) < unsigned(b); }

/*! Im2Col2d <T = float32, Device = CPU> */

template<typename T>
void _Im2Col2d_NCHW(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
    const int im_offset = H * W;
    for (int c = 0; c < C; ++c, im += im_offset) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = -pad_h + kh * dilation_h;
                for (int output_h = 0; output_h < col_h; ++output_h) {
                    if (!less(h, H)) {
                        for (int output_w = 0; output_w < col_w; ++output_w) *(col++) = 0;
                    } else {
                        int w = -pad_w + kw * dilation_w;
                        for (int output_w = 0; output_w < col_w; ++output_w) {
                            if (!less(w, W)) *(col++) = 0;
                            else *(col++) = im[h * W + w];
                            w += stride_w;
                        }
                    }
                    h += stride_h;
                }
            }
        }
    }
}

template<typename T>
void _Im2Col2d_NHWC(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
    for (int output_h = 0; output_h < col_h; ++output_h) {
        const int base_h = -pad_h + stride_h * output_h;
        for (int output_w = 0; output_w < col_w; ++output_w) {
            const int base_w = -pad_w + stride_w * output_w;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h = base_h + kh * dilation_h;
                if (!less(h, H)) {
                    for (int kw = 0; kw < kernel_w; ++kw)
                        for (int c = 0; c < C; ++c) *(col++) = 0;
                } else {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int w = base_w + kw * dilation_w;
                        for (int c = 0; c < C; ++c) {
                            if (!less(w, W)) *(col++) = 0;
                            else *(col++) = im[(h * W + w) * C + c];
                        }
                    }
                }
            }
        }
    }
}

template <> void Im2Col2d<float, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            im,
    float*                  col,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        const int count = (C * col_h * col_w);
        _Im2Col2d_NCHW<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, im, col);
    } else if (data_format == "NHWC") {
        const int count = (col_h * col_w * C);
        _Im2Col2d_NHWC<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, im, col);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! Col2Im2d <T = float32, Device = CPU> */

template<typename T>
void _Col2Im2d_NCHW(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im,
    CPUContext*             ctx) {
    math::Set<float, CPUContext>(C * H * W, 0, im, ctx);
    const int im_offset = H * W;
    for (int c = 0; c < C; ++c, im += im_offset) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = -pad_h + kh * dilation_h;
                for (int output_h = 0; output_h < col_h; ++output_h) {
                    if (!less(h, H)) {
                        col += col_w;
                    } else {
                        int w = -pad_w + kw * dilation_w;
                        for (int output_w = 0; output_w < col_w; ++output_w) {
                            if (less(w, W)) im[h * W + w] += *col;
                            ++col;
                            w += stride_w;
                        }
                    }
                    h += stride_h;
                }
            }
        } 
    }
}

template<typename T>
void _Col2Im2d_NHWC(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im,
    CPUContext*             ctx) {
    math::Set<float, CPUContext>(C * H * W, 0, im, ctx);
    for (int output_h = 0; output_h < col_h; ++output_h) {
        const int base_h = -pad_h + stride_h * output_h;
        for (int output_w = 0; output_w < col_w; ++output_w) {
            const int base_w = -pad_w + stride_w * output_w;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h = base_h + kh * dilation_h;
                if (!less(h, H)) {
                    col += (kernel_w * C);
                } else {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int w = base_w + kw * dilation_w;
                        for (int c = 0; c < C; ++c) {
                            if (less(w, W)) im[(h * W + w) * C + c] += *(col);
                            ++col;
                        }
                    }
                }
            }
        }
    }
}

template<> void Col2Im2d<float, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            col,
    float*                  im,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        const int count = (C * H * W);
        _Col2Im2d_NCHW<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, col, im, ctx);
    } else if (data_format == "NHWC") {
        const int count = (H * W * C);
        _Col2Im2d_NHWC<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, col, im, ctx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon