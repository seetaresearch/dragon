#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template <typename T>
void _MaxPool2dNCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    int*                    mask,
    T*                      y) {
    auto x_ofs = H * W, y_ofs = pool_h * pool_w;
    auto X_ofs = C * x_ofs, Y_ofs = C * y_ofs;
    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_ofs;
        auto* Y = y + n * Y_ofs;
        auto* M = mask + n * Y_ofs;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int h_start = ph * stride_h - pad_h;
                    int w_start = pw * stride_w - pad_w;
                    int h_end = std::min(h_start + kernel_h, H);
                    int w_end = std::min(w_start + kernel_w, W);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    const int yi = ph * pool_w + pw;
                    int max_idx = -1; T max_val = -FLT_MAX;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            const int xi = h * W + w;
                            if (X[xi] > max_val) {
                                max_idx = xi;
                                max_val = X[xi];
                            }
                        }
                    }
                    Y[yi] = max_val;
                    M[yi] = max_idx;
                }
            } 
            X += x_ofs;
            Y += y_ofs;
            M += y_ofs;
        }
    }
}

template <typename T>
void _MaxPool2dNHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    int*                    mask,
    T*                      y) {
    auto X_ofs = H * W * C;
    auto Y_ofs = pool_h * pool_w * C;
    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_ofs;
        auto* Y = y + n * Y_ofs;
        auto* M = mask + n * Y_ofs;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int h_start = ph * stride_h - pad_h;
                int w_start = pw * stride_w - pad_w;
                int h_end = std::min(h_start + kernel_h, H);
                int w_end = std::min(w_start + kernel_w, W);
                h_start = std::max(h_start, 0);
                w_start = std::max(w_start, 0);
                const int base_yi = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int yi = base_yi * C + c;
                    int max_idx = -1; T max_val = -FLT_MAX;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            const int xi = (h * W + w) * C + c;
                            if (X[xi] > max_val) {
                                max_idx = xi;
                                max_val = X[xi];
                            }
                        }
                    }
                    Y[yi] = max_val;
                    M[yi] = max_idx;
                }
            }
        }
    }
}

template<> void MaxPool2d<float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    int*                    mask,
    float*                  y,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _MaxPool2dNCHW(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, mask, y
        );
    } else if (data_format == "NHWC") {
        _MaxPool2dNHWC(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, mask, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template<typename T>
void _AvgPool2dNCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    T*                      y) {
    auto x_ofs = H * W, y_ofs = pool_h * pool_w;
    auto X_ofs = C * x_ofs, Y_ofs = C * y_ofs;
    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_ofs;
        auto* Y = y + n * Y_ofs;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int h_start = ph * stride_h - pad_h;
                    int w_start = pw * stride_w - pad_w;
                    int h_end = std::min(h_start + kernel_h, H + pad_h);
                    int w_end = std::min(w_start + kernel_w, W + pad_w);
                    T area = (h_end - h_start) * (w_end - w_start);
                    h_end = std::min(h_end, H);
                    w_end = std::min(w_end, W);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    const int yi = ph * pool_w + pw;
                    T sum_val = 0;
                    for (int h = h_start; h < h_end; ++h)
                        for (int w = w_start; w < w_end; ++w)
                            sum_val += X[h * W + w];
                    Y[yi] = sum_val / area;
                } 
            }
            X += x_ofs;
            Y += y_ofs;
        }
    }
}

template<typename T>
void _AvgPool2dNHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                x,
    T*                      y) {
    auto X_ofs = H * W * C;
    auto Y_ofs = pool_h * pool_w * C;
    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_ofs;
        auto* Y = y + n * Y_ofs;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int h_start = ph * stride_h - pad_h;
                int w_start = pw * stride_w - pad_w;
                int h_end = std::min(h_start + kernel_h, H + pad_h);
                int w_end = std::min(w_start + kernel_w, W + pad_w);
                T area = (h_end - h_start) * (w_end - w_start);
                h_end = std::min(h_end, H);
                w_end = std::min(w_end, W);
                h_start = std::max(h_start, 0);
                w_start = std::max(w_start, 0);
                const int base_yi = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int yi = base_yi * C + c;
                    T sum_val = 0;
                    for (int h = h_start; h < h_end; ++h)
                        for (int w = w_start; w < w_end; ++w)
                            sum_val += X[(h * W + w) * C + c];
                    Y[yi] = sum_val / area;
                }
            }
        }
    }
}

template<> void AvgPool2d<float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _AvgPool2dNCHW(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w, 
            x, y
        );
    } else if (data_format == "NHWC") {
        _AvgPool2dNHWC(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            x, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template <typename T>
void _MaxPool2dGradNCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const int*              mask,
    T*                      dx,
    CPUContext*             ctx) { 
    auto x_ofs = H * W, y_ofs = pool_h * pool_w;
    auto X_ofs = C * x_ofs, Y_ofs = C * y_ofs;
    math::Set(N * C * H * W, cast::to<T>(0.f), dx, ctx);
    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * Y_ofs;
        auto* dX = dx + n * X_ofs;
        auto* M = mask + n * Y_ofs;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    const int yi = ph * pool_w + pw;
                    const int xi = M[yi];
                    dX[xi] += dY[yi];
                }
            }
            dX += x_ofs;
            dY += y_ofs;
            M += y_ofs;
        }
    }
}

template <typename T>
void _MaxPool2dGradNHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    const int*              mask,
    T*                      dx,
    CPUContext*             ctx) {
    auto X_ofs = H * W * C;
    auto Y_ofs = pool_h * pool_w * C;
    math::Set(N * H * W * C, cast::to<T>(0.f), dx, ctx);
    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * X_ofs;
        auto* dX = dx + n * Y_ofs;
        auto* M = mask + n * Y_ofs;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                const int base_yi = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int yi = base_yi * C + c;
                    const int xi = M[yi];
                    dX[xi] += dY[yi];
                }
            }
        }
    }
}

template<> void MaxPool2dGrad<float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    const int*              mask,
    float*                  dx,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _MaxPool2dGradNCHW(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w, dy,
            mask, dx, ctx
        );
    } else if (data_format == "NHWC") {
        _MaxPool2dGradNHWC(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy, mask, dx, ctx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

/* <T = float32, Device = CPU> */

template <typename T>
void _AvgPool2dGradNCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    auto x_ofs = H * W, y_ofs = pool_h * pool_w;
    auto X_ofs = C * x_ofs, Y_ofs = C * y_ofs;
    math::Set(N * C * H * W, cast::to<T>(0.f), dx, ctx);
    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * Y_ofs;
        auto* dX = dx + n * X_ofs;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int h_start = ph * stride_h - pad_h;
                    int w_start = pw * stride_w - pad_w;
                    int h_end = std::min(h_start + kernel_h, H + pad_h);
                    int w_end = std::min(w_start + kernel_w, W + pad_w);
                    T area = (h_end - h_start) * (w_end - w_start);
                    h_end = std::min(h_end, H);
                    w_end = std::min(w_end, W);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    const int yi = ph * pool_w + pw;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            const int xi = h * W + w;
                            dX[xi] += dY[yi] / area;
                        }
                    }
                }
            } 
            dX += x_ofs;
            dY += y_ofs;
        }
    }
}

template <typename T>
void _AvgPool2dGradNHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    auto X_ofs = H * W * C;
    auto Y_ofs = pool_h * pool_w * C;
    math::Set(N * H * W * C, cast::to<T>(0.f), dx, ctx);
    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * X_ofs;
        auto* dX = dx + n * Y_ofs;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int h_start = ph * stride_h - pad_h;
                int w_start = pw * stride_w - pad_w;
                int h_end = std::min(h_start + kernel_h, H + pad_h);
                int w_end = std::min(w_start + kernel_w, W + pad_w);
                T area = (h_end - h_start) * (w_end - w_start);
                h_end = std::min(h_end, H);
                w_end = std::min(w_end, W);
                h_start = std::max(h_start, 0);
                w_start = std::max(w_start, 0);
                const int base_yi = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int yi = base_yi * C + c;
                    for (int h = h_start; h < h_end; ++h)
                        for (int w = w_start; w < w_end; ++w)
                            dX[(h * W + w) * C + c] += dY[yi] / area;
                }
            }
        }
    }
}

template<> void AvgPool2dGrad<float, CPUContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    if (data_format == "NCHW") {
        _AvgPool2dGradNCHW(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy, dx, ctx
        );
    } else if (data_format == "NHWC") {
        _AvgPool2dGradNHWC(
            N, C, H, W,
            pool_h, pool_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dy, dx, ctx
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon