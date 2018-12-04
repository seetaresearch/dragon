#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! BilinearResize <T = float32, Device = CPU> */

template <typename T>
void _BilinearResize_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int NC = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const float h_in = h * scale_h;
                const int top_y_idx = (int)floorf(h_in);
                const int bottom_y_idx = (h_in < H - 1) ? (int)ceilf(h_in) : H - 1;
                const int NCHT = NC * H + top_y_idx;
                const int NCHB = NC * H + bottom_y_idx;
                const float y_lerp = h_in - top_y_idx;
                for (int w = 0; w < out_w; ++w) {
                    const float w_in = w * scale_w;
                    const int left_x_idx = (int)floorf(w_in);
                    const int right_x_idx = (w_in < W - 1) ? (int)ceilf(w_in) : W - 1;
                    const float x_lerp = w_in - left_x_idx;

                    const float top_left(x[NCHT * W + left_x_idx]);
                    const float top_right(x[NCHT * W + right_x_idx]);
                    const float bottom_left(x[NCHB * W + left_x_idx]);
                    const float bottom_right(x[NCHB * W + right_x_idx]);

                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                    *(y++) = top + (bottom - top) * y_lerp;
                }
            }
        }
    }
}

template <typename T>
void _BilinearResize_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_h; ++h) {
            const float h_in = h * scale_h;
            const int top_y_idx = (int)floorf(h_in);
            const int bottom_y_idx = (h_in < H - 1) ? (int)ceilf(h_in) : H - 1;
            const int NHT = n * H + top_y_idx;
            const int NHB = n * H + bottom_y_idx;
            const float y_lerp = h_in - top_y_idx;
            for (int w = 0; w < out_w; ++w) {
                const float w_in = w * scale_w;
                const int left_x_idx = (int)floorf(w_in);
                const int right_x_idx = (w_in < W - 1) ? (int)ceilf(w_in) : W - 1;
                const float x_lerp = w_in - left_x_idx;
                for (int c = 0; c < C; ++c) {
                    const float top_left(x[(NHT * W + left_x_idx) * C + c]);
                    const float top_right(x[(NHT * W + right_x_idx) * C + c]);
                    const float bottom_left(x[(NHB * W + left_x_idx) * C + c]);
                    const float bottom_right(x[(NHB * W + right_x_idx) * C + c]);
                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                    *(y++) = top + (bottom - top) * y_lerp;
                }
            }
        }
    }
}

template <> void BilinearResize<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _BilinearResize_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if (data_format == "NHWC"){
        _BilinearResize_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! BilinearResizeGrad <T = float32, Device = CPU> */

template <typename T>
void _BilinearResizeGrad_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int NC = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const float h_in = h * scale_h;
                const int top_y_idx = (int)floorf(h_in);
                const int bottom_y_idx = (h_in < H - 1) ? (int)ceilf(h_in) : H - 1;
                const int NCHT = NC * H + top_y_idx;
                const int NCHB = NC * H + bottom_y_idx;
                const float y_lerp = h_in - top_y_idx;
                for (int w = 0; w < out_w; ++w) {
                    const float w_in = w * scale_w;
                    const int left_x_idx = (int)floorf(w_in);
                    const int right_x_idx = (w_in < W - 1) ? (int)ceilf(w_in) : W - 1;
                    const float x_lerp = w_in - left_x_idx;
                    const float dtop = (1 - y_lerp) * (*(dy));
                    const float dbottom = y_lerp * (*(dy++));
                    dx[NCHT * W + left_x_idx] +=
                        static_cast<T>((1 - x_lerp) * dtop);
                    dx[NCHT * W + right_x_idx] +=
                        static_cast<T>(x_lerp * dtop);
                    dx[NCHB * W + left_x_idx] +=
                        static_cast<T>((1 - x_lerp) * dbottom);
                    dx[NCHB * W + right_x_idx] += 
                        static_cast<T>(x_lerp * dbottom);
                }
            }
        }
    }
}

template <typename T>
void _BilinearResizeGrad_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_h; ++h) {
            const float h_in = h * scale_h;
            const int top_y_idx = (int)floorf(h_in);
            const int bottom_y_idx = (h_in < H - 1) ? (int)ceilf(h_in) : H - 1;
            const int NHT = n * H + top_y_idx;
            const int NHB = n * H + bottom_y_idx;
            const float y_lerp = h_in - top_y_idx;
            for (int w = 0; w < out_w; ++w) {
                const float w_in = w * scale_w;
                const int left_x_idx = (int)floorf(w_in);
                const int right_x_idx = (w_in < W - 1) ? (int)ceilf(w_in) : W - 1;
                const float x_lerp = w_in - left_x_idx;
                const float dtop = (1 - y_lerp) * (*(dy));
                const float dbottom = y_lerp * (*(dy++));
                for (int c = 0; c < C; ++c) {
                    dx[(NHT * W + left_x_idx) * C + c] +=
                        static_cast<T>((1 - x_lerp) * dtop);
                    dx[(NHT * W + right_x_idx) * C + c] +=
                        static_cast<T>(x_lerp * dtop);
                    dx[(NHB * W + left_x_idx) * C + c] +=
                        static_cast<T>((1 - x_lerp) * dbottom);
                    dx[(NHB * W + right_x_idx) * C + c] += 
                        static_cast<T>(x_lerp * dbottom);
                }
            }
        }
    }
}

template <> void BilinearResizeGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _BilinearResizeGrad_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else if (data_format == "NHWC"){
        _BilinearResizeGrad_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon