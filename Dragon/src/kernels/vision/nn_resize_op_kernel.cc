#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! NNResize <T = ?, Device = CPU> */

template <typename T>
void _NNResize_NCHW(
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
                const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
                const int NCH = NC * H + h_in;
                for (int w = 0; w < out_w; ++w) {
                    const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                    *(y++) = x[NCH * W + w_in];
                }
            }
        }
    }
}

template <typename T>
void _NNResize_NHWC(
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
            const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
            const int NH = n * H + h_in;
            for (int w = 0; w < out_w; ++w) {
                const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                const int NHW = NH * W + w_in;
                for (int c = 0; c < C; ++c) *(y++) = x[NHW * C + c];
            }
        }
    }
}

/*! NNResize <T = float32, Device = CPU> */

template <> void NNResize<float, CPUContext>(
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
        _NNResize_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if (data_format == "NHWC"){
        _NNResize_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! NNResize <T = float16, Device = CPU> */

template <> void NNResize<float16, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<float16>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if (data_format == "NHWC"){
        _NNResize_NHWC<float16>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! NNResizeGrad <T = float32, Device = CPU> */

template <typename T>
void _NNResizeGrad_NCHW(
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
                const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
                const int NCH = NC * H + h_in;
                for (int w = 0; w < out_w; ++w) {
                    const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                    dx[NCH * W + w_in] += *(dy++);
                }
            }
        }
    }
}

template <typename T>
void _NNResizeGrad_NHWC(
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
            const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
            const int NH = n * H + h_in;
            for (int w = 0; w < out_w; ++w) {
                const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                const int NHW = NH * W + w_in;
                for (int c = 0; c < C; ++c) dx[NHW * C + c] += *(dy++);
            }
        }
    }
}

template <> void NNResizeGrad<float, CPUContext>(
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
        _NNResizeGrad_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else if (data_format == "NHWC"){
        _NNResizeGrad_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon