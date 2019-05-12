#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template <typename T>
void _BilinearResizeNCHW(
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
                const float h_in = h * scale_h;
                const int tyi = (int)floorf(h_in);
                const int byi = (h_in < H - 1) ?
                    (int)ceilf(h_in) : H - 1;
                const int ncht = nc * H + tyi;
                const int nchb = nc * H + byi;
                const T ylerp = h_in - tyi;
                for (int w = 0; w < out_w; ++w) {
                    const float w_in = w * scale_w;
                    const int lxi = (int)floorf(w_in);
                    const int rxi = (w_in < W - 1) ?
                        (int)ceilf(w_in) : W - 1;
                    const T xlerp = w_in - lxi;
                    const T tl(x[ncht * W + lxi]);
                    const T tr(x[ncht * W + rxi]);
                    const T bl(x[nchb * W + lxi]);
                    const T br(x[nchb * W + rxi]);
                    const T t = tl + (tr - tl) * xlerp;
                    const T b = bl + (br - bl) * xlerp;
                    *(y++) = t + (b - t) * ylerp;
                }
            }
        }
    }
}

template <typename T>
void _BilinearResizeNHWC(
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
            const int tyi = (int)floorf(h_in);
            const int byi = (h_in < H - 1) ? 
                (int)ceilf(h_in) : H - 1;
            const int nht = n * H + tyi;
            const int nhb = n * H + byi;
            const T ylerp = h_in - tyi;
            for (int w = 0; w < out_w; ++w) {
                const float w_in = w * scale_w;
                const int lxi = (int)floorf(w_in);
                const int rxi = (w_in < W - 1) ?
                    (int)ceilf(w_in) : W - 1;
                const T xlerp = w_in - lxi;
                for (int c = 0; c < C; ++c) {
                    const T tl(x[(nht * W + lxi) * C + c]);
                    const T tr(x[(nht * W + rxi) * C + c]);
                    const T bl(x[(nhb * W + lxi) * C + c]);
                    const T br(x[(nhb * W + rxi) * C + c]);
                    const T t = tl + (tr - tl) * xlerp;
                    const T b = bl + (br - bl) * xlerp;
                    *(y++) = t + (b - t) * ylerp;
                }
            }
        }
    }
}

template <> void BilinearResize<float, CPUContext>(
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
        _BilinearResizeNCHW(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, x, y
        );
    } else if (data_format == "NHWC"){
        _BilinearResizeNHWC(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, x, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template <typename T>
void _BilinearResizeGradNCHW(
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
                const int tyi = (int)floorf(h_in);
                const int byi = (h_in < H - 1) ?
                    (int)ceilf(h_in) : H - 1;
                const int ncht = NC * H + tyi;
                const int nchb = NC * H + byi;
                const T ylerp = h_in - tyi;
                for (int w = 0; w < out_w; ++w) {
                    const float w_in = w * scale_w;
                    const int lxi = (int)floorf(w_in);
                    const int rxi = (w_in < W - 1) ?
                        (int)ceilf(w_in) : W - 1;
                    const T xlerp = w_in - lxi;
                    const T dt = (T(1) - ylerp) * (*(dy));
                    const T db = ylerp * (*(dy++));
                    dx[ncht * W + lxi] += (T(1) - xlerp) * dt;
                    dx[ncht * W + rxi] += xlerp * dt;
                    dx[nchb * W + lxi] += (T(1) - xlerp) * db;
                    dx[nchb * W + rxi] += xlerp * db;
                }
            }
        }
    }
}

template <typename T>
void _BilinearResizeGradNHWC(
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
            const int tyi = (int)floorf(h_in);
            const int byi = (h_in < H - 1) ?
                (int)ceilf(h_in) : H - 1;
            const int nht = n * H + tyi;
            const int nhb = n * H + byi;
            const T ylerp = h_in - tyi;
            for (int w = 0; w < out_w; ++w) {
                const float w_in = w * scale_w;
                const int lxi = (int)floorf(w_in);
                const int rxi = (w_in < W - 1) ?
                    (int)ceilf(w_in) : W - 1;
                const T xlerp = w_in - lxi;
                const T dt = (T(1) - ylerp) * (*(dy));
                const T db = ylerp * (*(dy++));
                for (int c = 0; c < C; ++c) {
                    dx[(nht * W + lxi) * C + c] += (T(1) - xlerp) * dt;
                    dx[(nht * W + rxi) * C + c] += xlerp * dt;
                    dx[(nhb * W + lxi) * C + c] += (T(1) - xlerp) * db;
                    dx[(nhb * W + rxi) * C + c] += xlerp * db;
                }
            }
        }
    }
}

template <> void BilinearResizeGrad<float, CPUContext>(
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
        _BilinearResizeGradNCHW(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, dy, dx
        );
    } else if (data_format == "NHWC"){
        _BilinearResizeGradNHWC(
            N, C, H, W, out_h, out_w,
            scale_h, scale_w, dy, dx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon