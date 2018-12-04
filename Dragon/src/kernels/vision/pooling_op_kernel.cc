#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/*! MAXPooling2d <T = float32, Device = CPU> */

template <typename T>
void _MAXPooling2d_NCHW(
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
    const float*            x,
    int*                    mask,
    float*                  y) {
    const TIndex x_offset = H * W, y_offset = pool_h * pool_w;
    const TIndex X_offset = C * x_offset, Y_offset = C * y_offset;

    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_offset;
        auto* Y = y + n * Y_offset;
        auto* M = mask + n * Y_offset;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, H);
                    int end_w = std::min(start_w + kernel_w, W);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_w + pw;
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * W + w;
                            if (X[idx] > max_val) {
                                max_val = X[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    Y[pool_idx] = max_val;
                    M[pool_idx] = max_idx;
                }
            } 
            X += x_offset;
            Y += y_offset;
            M += y_offset;
        }
    }
}

template <typename T>
void _MAXPooling2d_NHWC(
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
    const float*            x,
    int*                    mask,
    float*                  y) {
    const TIndex X_offset = H * W * C,
        Y_offset = pool_h * pool_w * C;

    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_offset;
        auto* Y = y + n * Y_offset;
        auto* M = mask + n * Y_offset;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = std::min(start_h + kernel_h, H);
                int end_w = std::min(start_w + kernel_w, W);
                start_h = std::max(start_h, 0);
                start_w = std::max(start_w, 0);
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = (h * W + w) * C + c;
                            if (X[idx] > max_val) {
                                max_val = X[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    Y[pool_idx] = max_val;
                    M[pool_idx] = max_idx;
                }
            }
        }
        x += X_offset;
        y += Y_offset;
        mask += Y_offset;
    }
}

template<> void MAXPooling2d<float, CPUContext>(
    const int               count,
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
        _MAXPooling2d_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else if (data_format == "NHWC") {
        _MAXPooling2d_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! AVGPooling2d <T = float32, Device = CPU> */

template<typename T>
void _AVGPooling2d_NCHW(
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
    const float*            x,
    float*                  y) {
    const TIndex x_offset = H * W, y_offset = pool_h * pool_w;
    const TIndex X_offset = C * x_offset, Y_offset = C * y_offset;

    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_offset;
        auto* Y = y + n * Y_offset;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, H + pad_h);
                    int end_w = std::min(start_w + kernel_w, W + pad_w);
                    int pool_area = (end_h - start_h) * (end_w - start_w);
                    end_h = std::min(end_h, H);
                    end_w = std::min(end_w, W);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_w + pw;
                    T sum_val = 0;
                    for (int h = start_h; h < end_h; ++h)
                        for (int w = start_w; w < end_w; ++w)
                            sum_val += X[h * W + w];
                    Y[pool_idx] = sum_val / pool_area;
                } 
            }
            X += x_offset;
            Y += y_offset;
        }
    }
}

template<typename T>
void _AVGPooling2d_NHWC(
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
    const float*            x,
    float*                  y) {
    const TIndex X_offset = H * W * C,
        Y_offset = pool_h * pool_w * C;

    for (int n = 0; n < N; ++n) {
        auto* X = x + n * X_offset;
        auto* Y = y + n * Y_offset;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = std::min(start_h + kernel_h, H + pad_h);
                int end_w = std::min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                end_h = std::min(end_h, H);
                end_w = std::min(end_w, W);
                start_h = std::max(start_h, 0);
                start_w = std::max(start_w, 0);
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    T sum_val = 0;
                    for (int h = start_h; h < end_h; ++h)
                        for (int w = start_w; w < end_w; ++w)
                            sum_val += X[(h * W + w) * C + c];
                    Y[pool_idx] = sum_val / pool_area;
                }
            }
        }
        X += X_offset;
        Y += Y_offset;
    }
}

template<> void AVGPooling2d<float, CPUContext>(
    const int               count,
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
        _AVGPooling2d_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else if (data_format == "NHWC") {
        _AVGPooling2d_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! MAXPooling2dGrad <T = float32, Device = CPU> */

template <typename T>
void _MAXPooling2dGrad_NCHW(
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
    const float*            dy,
    const int*              mask,
    float*                  dx,
    CPUContext*             ctx) {
    const TIndex x_offset = H * W, y_offset = pool_h * pool_w;
    const TIndex X_offset = C * x_offset, Y_offset = C * y_offset;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx, ctx);

    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * Y_offset;
        auto* dX = dx + n * X_offset;
        auto* M = mask + n * Y_offset;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    const int pool_idx = ph * pool_w + pw;
                    const int idx = M[pool_idx];
                    dX[idx] += dY[pool_idx];
                }
            }
            dX += x_offset;
            dY += y_offset;
            M += y_offset;
        }
    }
}

template <typename T>
void _MAXPooling2dGrad_NHWC(
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
    const float*            dy,
    const int*              mask,
    float*                  dx,
    CPUContext*             ctx) {
    const TIndex X_offset = H * W * C,
        Y_offset = pool_h * pool_w * C;
    math::Set<float, CPUContext>(N * H * W * C, 0, dx, ctx);

    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * X_offset;
        auto* dX = dx + n * Y_offset;
        auto* M = mask + n * Y_offset;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    const int idx = M[pool_idx];
                    dX[idx] += dY[pool_idx];
                }
            }
        }
        dX += X_offset;
        dY += Y_offset;
    }
}

template<> void MAXPooling2dGrad<float, CPUContext>(
    const int               count,
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
        _MAXPooling2dGrad_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, mask, dx, ctx);
    } else if (data_format == "NHWC") {
        _MAXPooling2dGrad_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, mask, dx, ctx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/*! AVGPooling2dGrad <T = float32, Device = CPU> */

template <typename T>
void _AVGPooling2dGrad_NCHW(
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
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    const TIndex x_offset = H * W, y_offset = pool_h * pool_w;
    const TIndex X_offset = C * x_offset, Y_offset = C * y_offset;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx, ctx);

    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * Y_offset;
        auto* dX = dx + n * X_offset;
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, H + pad_h);
                    int end_w = std::min(start_w + kernel_w, W + pad_w);
                    int pool_area = (end_h - start_h) * (end_w - start_w);
                    end_h = std::min(end_h, H);
                    end_w = std::min(end_w, W);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_w + pw;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * W + w;
                            dX[idx] += (dY[pool_idx] / pool_area);
                        }
                    }
                }
            } 
            dX += x_offset;
            dY += y_offset;
        }
    }
}

template <typename T>
void _AVGPooling2dGrad_NHWC(
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
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    const TIndex X_offset = H * W * C,
        Y_offset = pool_h * pool_w * C;
    math::Set<float, CPUContext>(N * H * W * C, 0, dx, ctx);

    for (int n = 0; n < N; ++n) {
        auto* dY = dy + n * X_offset;
        auto* dX = dx + n * Y_offset;
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = std::min(start_h + kernel_h, H + pad_h);
                int end_w = std::min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                end_h = std::min(end_h, H);
                end_w = std::min(end_w, W);
                start_h = std::max(start_h, 0);
                start_w = std::max(start_w, 0);
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    for (int h = start_h; h < end_h; ++h)
                        for (int w = start_w; w < end_w; ++w)
                            dX[(h * W + w) * C + c] +=
                                (dY[pool_idx] / pool_area);
                }
            }
        }
        dX += X_offset;
        dY += Y_offset;
    }
}

template<> void AVGPooling2dGrad<float, CPUContext>(
    const int               count,
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
        _AVGPooling2dGrad_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx, ctx);
    } else if (data_format == "NHWC") {
        _AVGPooling2dGrad_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx, ctx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon