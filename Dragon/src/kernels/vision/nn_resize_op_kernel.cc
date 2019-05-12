#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _NNResizeNCHW(
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
            const int nc = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const int h_in = std::min(
                    int(floorf(h * scale_h)), H - 1);
                const int nch = nc * H + h_in;
                for (int w = 0; w < out_w; ++w) {
                    const int w_in = std::min(
                        int(floorf(w * scale_w)), W - 1);
                    *(y++) = x[nch * W + w_in];
                }
            }
        }
    }
}

template <typename T>
void _NNResizeNHWC(
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
            const int h_in = std::min(
                int(floorf(h * scale_h)), H - 1);
            const int nh = n * H + h_in;
            for (int w = 0; w < out_w; ++w) {
                const int w_in = std::min(
                    int(floorf(w * scale_w)), W - 1);
                const int nhw = nh * W + w_in;
                for (int c = 0; c < C; ++c)
                    *(y++) = x[nhw * C + c];
            }
        }
    }
}

/* <T = float32, Device = CPU> */

template <> void NNResize<float, CPUContext>(
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
    auto scale_h = (float)H / (float)out_h;
    auto scale_w = (float)W / (float)out_w;
    if (data_format == "NCHW") {
        _NNResizeNCHW<float>(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, x, y
        );
    } else if (data_format == "NHWC"){
        _NNResizeNHWC(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, x, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float16, Device = CPU> */

template <> void NNResize<float16, CPUContext>(
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
        _NNResizeNCHW<float16>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if (data_format == "NHWC"){
        _NNResizeNHWC<float16>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template <typename T>
void _NNResizeGradNCHW(
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
            const int nc = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const int h_in = std::min(
                    int(floorf(h * scale_h)), H - 1);
                const int nch = nc * H + h_in;
                for (int w = 0; w < out_w; ++w) {
                    const int w_in = std::min(
                        int(floorf(w * scale_w)), W - 1);
                    dx[nch * W + w_in] += *(dy++);
                }
            }
        }
    }
}

template <typename T>
void _NNResizeGradNHWC(
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
            const int h_in = std::min(
                int(floorf(h * scale_h)), H - 1);
            const int nh = n * H + h_in;
            for (int w = 0; w < out_w; ++w) {
                const int w_in = std::min(
                    int(floorf(w * scale_w)), W - 1);
                const int nhw = nh * W + w_in;
                for (int c = 0; c < C; ++c)
                    dx[nhw * C + c] += *(dy++);
            }
        }
    }
}

template <> void NNResizeGrad<float, CPUContext>(
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
    auto scale_h = (float)H / (float)out_h;
    auto scale_w = (float)W / (float)out_w;
    if (data_format == "NCHW") {
        _NNResizeGradNCHW(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, dy, dx
        );
    } else if (data_format == "NHWC"){
        _NNResizeGradNHWC(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, dy, dx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon